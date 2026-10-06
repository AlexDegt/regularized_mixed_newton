"""
Защита от CUDA OOM на общем сервере.

Идея: вычислительный кусок (например, одна эпоха) оборачивается в OOMGuard.run().
Если внутри случается нехватка памяти GPU, то:
  1) исключение перехватывается, все тензоры неудачной попытки освобождаются;
  2) вызывается on_oom() -- например, откат параметров модели к состоянию до попытки;
  3) процесс ждёт, пока на GPU освободится достаточно памяти (опрос torch.cuda.mem_get_info);
  4) попытка повторяется.

Сколько памяти ждать, оценивается по пиковому потреблению прошлых успешных попыток
(и по сообщению "Tried to allocate X GiB" из текста ошибки).
"""
import gc
import re
import time
from typing import Callable, Optional

import torch

_OOM_MARKERS = (
    "out of memory",
    "cublas_status_alloc_failed",
    "cusolver_status_alloc_failed",
    "cudnn_status_alloc_failed",
)
_TRIED_RE = re.compile(r"Tried to allocate ([\d.]+) ([KMGT]i?B)")
_UNITS = {"KiB": 2 ** 10, "MiB": 2 ** 20, "GiB": 2 ** 30, "TiB": 2 ** 40,
          "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9, "TB": 10 ** 12}


def is_oom_error(exc: BaseException) -> bool:
    """True, если исключение -- нехватка памяти GPU (в т.ч. внутри cuBLAS/cuSOLVER)."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(m in msg for m in _OOM_MARKERS)
    return False


def _tried_to_allocate(msg: str) -> int:
    m = _TRIED_RE.search(msg)
    if m is None:
        return 0
    return int(float(m.group(1)) * _UNITS.get(m.group(2), 1))


def _gib(n_bytes: float) -> str:
    return f"{n_bytes / 2 ** 30:.2f} GiB"


def free_cuda_memory() -> None:
    """Вернуть драйверу всю кэшированную, но неиспользуемую память."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class OOMGuard:
    """
    Args:
        device: GPU, на котором считаем (torch.device или индекс).
        poll_interval: как часто (с) проверять свободную память во время ожидания.
        max_wait: максимальное время ожидания одного освобождения памяти (с); None -- ждать сколько угодно.
        max_retries: максимум повторов одного вызова run(); None -- без ограничения.
        safety_factor: запас к оценке требуемой памяти.
        extra_bytes: дополнительный фиксированный запас.
        min_wait: минимальная пауза после OOM, даже если память формально уже свободна
            (чужой процесс мог её лишь на мгновение отпустить).
        log: функция логирования.
    """

    def __init__(self, device=None, poll_interval: float = 60.0, max_wait: Optional[float] = None,
                 max_retries: Optional[int] = None, safety_factor: float = 1.1,
                 extra_bytes: int = 256 * 2 ** 20, min_wait: float = 30.0,
                 log: Callable[[str], None] = print):
        self.device = torch.device(device) if device is not None else torch.device("cuda")
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.max_retries = max_retries
        self.safety_factor = safety_factor
        self.extra_bytes = extra_bytes
        self.min_wait = min_wait
        self.log = log
        # пик зарезервированной памяти (абсолютный, в байтах) успешных попыток, по ключам
        self.peak_needed = {}
        # статистика: время последней успешной попытки и время ожидания в последнем run()
        self.last_run_time = 0.0
        self.last_wait_time = 0.0
        self.total_wait_time = 0.0
        self.oom_count = 0

    # ---------- ожидание памяти ----------
    def _need_bytes(self, key: str, peak_failed: int = 0, tried: int = 0) -> int:
        target_peak = max(self.peak_needed.get(key, 0), peak_failed + tried)
        held = torch.cuda.memory_reserved(self.device)
        return max(int(target_peak * self.safety_factor) - held, 0) + self.extra_bytes

    def _wait_for_memory(self, need_bytes: int, min_wait: float = 0.0) -> float:
        start = time.time()
        if min_wait > 0:
            time.sleep(min_wait)
        last_report = float("-inf")
        while True:
            free, total = torch.cuda.mem_get_info(self.device)
            if free >= need_bytes:
                waited = time.time() - start
                self.log(f"[OOMGuard] свободно {_gib(free)} из {_gib(total)}, "
                         f"нужно ~{_gib(need_bytes)}; продолжаем (ждали {waited:.0f} с)")
                return waited
            elapsed = time.time() - start
            if self.max_wait is not None and elapsed > self.max_wait:
                raise TimeoutError(f"[OOMGuard] память не освободилась за {self.max_wait} с "
                                   f"(свободно {_gib(free)}, нужно {_gib(need_bytes)})")
            if elapsed - last_report >= 600:  # напоминание в лог раз в 10 минут
                self.log(f"[OOMGuard] ждём память: свободно {_gib(free)} из {_gib(total)}, "
                         f"нужно ~{_gib(need_bytes)}, прошло {elapsed:.0f} с")
                last_report = elapsed
            time.sleep(self.poll_interval)

    # ---------- основной метод ----------
    def run(self, fn: Callable, key: str = "default", on_oom: Optional[Callable[[], None]] = None,
            precheck: bool = True):
        """
        Выполнить fn() с повтором при OOM.

        key: имя вычислительного блока; для каждого ключа запоминается пиковая память,
            чтобы знать, сколько ждать при следующем OOM.
        on_oom: вызывается после каждой неудачной попытки (после освобождения памяти),
            должен вернуть состояние к тому, что было до fn() (например, параметры модели).
        precheck: если пик для key уже известен, перед попыткой дождаться нужного объёма
            свободной памяти, чтобы не тратить время на заведомо неудачную попытку.
        """
        attempt = 0
        self.last_wait_time = 0.0

        if precheck and key in self.peak_needed:
            need = self._need_bytes(key)
            free, _ = torch.cuda.mem_get_info(self.device)
            if free < need:
                waited = self._wait_for_memory(need)
                self.last_wait_time += waited
                self.total_wait_time += waited

        while True:
            torch.cuda.reset_peak_memory_stats(self.device)
            t0 = time.time()
            oom_msg = None
            try:
                result = fn()
                self.last_run_time = time.time() - t0
                peak = torch.cuda.max_memory_reserved(self.device)
                self.peak_needed[key] = max(self.peak_needed.get(key, 0), peak)
                return result
            except Exception as exc:  # noqa: BLE001
                if not is_oom_error(exc):
                    raise
                oom_msg = str(exc)
            # Мы вне блока except: исключение и его traceback (со всеми тензорами
            # неудачной попытки в кадрах стека) уже не держатся -- можно чистить память.
            attempt += 1
            self.oom_count += 1
            peak_failed = torch.cuda.max_memory_reserved(self.device)
            tried = _tried_to_allocate(oom_msg)
            free_cuda_memory()

            first_line = oom_msg.strip().splitlines()[0][:200] if oom_msg else ""
            self.log(f"[OOMGuard] OOM в блоке '{key}' (попытка {attempt}): {first_line}")
            if self.max_retries is not None and attempt > self.max_retries:
                raise RuntimeError(f"[OOMGuard] превышено число повторов ({self.max_retries}) для '{key}'")

            need = self._need_bytes(key, peak_failed, tried)
            waited = self._wait_for_memory(need, min_wait=self.min_wait)
            self.last_wait_time += waited
            self.total_wait_time += waited

            if on_oom is not None:
                # сам откат тоже может упасть по памяти -- повторяем его до успеха
                while True:
                    try:
                        on_oom()
                        break
                    except Exception as exc:  # noqa: BLE001
                        if not is_oom_error(exc):
                            raise
                    free_cuda_memory()
                    time.sleep(self.poll_interval)
