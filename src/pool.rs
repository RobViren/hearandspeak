use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, Thread};

// Calls arrive back to back (hundreds per ASR segment), so a futex sleep between them costs more than the spin.
const SPIN: u32 = 3000;

type Job = &'static (dyn Fn(usize) + Sync);

struct Shared {
    generation: AtomicU64,
    pending: AtomicUsize,
    stop: AtomicBool,
    job: UnsafeCell<Option<Job>>,
    caller: UnsafeCell<Option<Thread>>,
}
unsafe impl Sync for Shared {}

/// Persistent workers pinned one per logical CPU: physical cores fastest first, then hyperthread siblings.
/// The calling thread takes part as worker 0. ASR and TTS share one pool; calls are serialized.
pub struct Pool {
    pub n: usize,
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    slow: Vec<bool>,
    fast: usize,
    lock: Mutex<()>,
}

impl Pool {
    /// `threads == 0` uses the fast physical cores (the P-cores on a hybrid CPU), where throughput peaks.
    pub fn new(threads: usize) -> Arc<Pool> {
        let cpus = cpu::fast_cores();
        let n = if threads > 0 {
            threads
        } else {
            cpus.iter().filter(|c| !c.slow && !c.sibling).count().max(1)
        };
        let cpu_of = |i: usize| (!cpus.is_empty()).then(|| cpus[i % cpus.len()]);
        let slow: Vec<bool> = (0..n).map(|i| cpu_of(i).is_some_and(|c| c.slow)).collect();
        let fast = slow.iter().filter(|s| !**s).count();
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            job: UnsafeCell::new(None),
            caller: UnsafeCell::new(None),
        });
        if let Some(c) = cpu_of(0) {
            cpu::pin(c.id);
        }
        let workers = (1..n)
            .map(|i| {
                let s = shared.clone();
                let id = cpu_of(i).map(|c| c.id);
                thread::Builder::new()
                    .name(format!("hs{i}"))
                    .spawn(move || work(&s, i, id))
                    .unwrap()
            })
            .collect();
        Arc::new(Pool {
            n,
            shared,
            workers,
            slow,
            fast,
            lock: Mutex::new(()),
        })
    }

    /// f(i) for i in 0..n, one call per worker, returning when all are done.
    pub fn run<F: Fn(usize) + Sync>(&self, f: F) {
        if self.n == 1 {
            return f(0);
        }
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let s = &*self.shared;
        let job: &(dyn Fn(usize) + Sync) = &f;
        unsafe {
            *s.job.get() = Some(std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(job));
            *s.caller.get() = Some(thread::current());
        }
        s.pending.store(self.n - 1, SeqCst);
        s.generation.fetch_add(1, SeqCst);
        for w in &self.workers {
            w.thread().unpark();
        }
        f(0);
        let mut spin = 0;
        while s.pending.load(Acquire) != 0 {
            if spin < SPIN {
                spin += 1;
                std::hint::spin_loop();
            } else {
                thread::park();
            }
        }
    }

    /// Work claimed in grains from a shared counter, so fast cores take more. Slow cores stop claiming once the
    /// fast ones can finish the rest in one grain each; otherwise every call ends waiting on an E-core.
    pub fn for_each<F: Fn(usize) + Sync>(&self, count: usize, grain: usize, f: F) {
        let grain = if grain == 0 {
            (count / (8 * self.n)).max(1)
        } else {
            grain
        };
        let next = AtomicUsize::new(0);
        self.run(|i| {
            loop {
                if self.slow[i] && count.saturating_sub(next.load(Relaxed)) <= grain * self.fast {
                    return;
                }
                let j = next.fetch_add(grain, Relaxed);
                if j >= count {
                    return;
                }
                for k in j..count.min(j + grain) {
                    f(k);
                }
            }
        });
    }

    /// Fixed contiguous split: worker i takes [count * i / n, count * (i + 1) / n).
    pub fn for_split<F: Fn(usize) + Sync>(&self, count: usize, f: F) {
        let n = self.n;
        self.run(|i| {
            for j in count * i / n..count * (i + 1) / n {
                f(j);
            }
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, SeqCst);
        self.shared.generation.fetch_add(1, SeqCst);
        for w in self.workers.drain(..) {
            w.thread().unpark();
            let _ = w.join();
        }
    }
}

fn work(s: &Shared, i: usize, cpu: Option<usize>) {
    if let Some(c) = cpu {
        cpu::pin(c);
    }
    let mut seen = 0;
    loop {
        let mut spin = 0;
        while s.generation.load(Acquire) == seen {
            if spin < SPIN {
                spin += 1;
                std::hint::spin_loop();
            } else {
                thread::park();
            }
        }
        seen = s.generation.load(Acquire);
        if s.stop.load(Acquire) {
            return;
        }
        let (job, caller) =
            unsafe { ((*s.job.get()).unwrap(), (*s.caller.get()).clone().unwrap()) };
        job(i);
        if s.pending.fetch_sub(1, AcqRel) == 1 {
            caller.unpark();
        }
    }
}

/// A raw pointer shared across pool workers that write disjoint parts of one buffer.
pub struct P<T>(pub *mut T);
impl<T> Clone for P<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for P<T> {}
unsafe impl<T> Send for P<T> {}
unsafe impl<T> Sync for P<T> {}
impl<T> P<T> {
    pub fn new(s: &mut [T]) -> Self {
        P(s.as_mut_ptr())
    }
    #[inline(always)]
    pub fn ptr(self) -> *mut T {
        self.0
    }
    #[inline(always)]
    pub fn at(self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice<'a>(self, i: usize, n: usize) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(i), n) }
    }
}

mod cpu {
    #[derive(Clone, Copy)]
    pub struct Cpu {
        pub id: usize,
        pub slow: bool,
        pub sibling: bool,
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    unsafe fn syscall3(n: usize, a: usize, b: usize, c: usize) -> isize {
        let ret: isize;
        unsafe {
            std::arch::asm!("syscall", inlateout("rax") n as isize => ret, in("rdi") a, in("rsi") b, in("rdx") c,
                lateout("rcx") _, lateout("r11") _, options(nostack));
        }
        ret
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    pub fn pin(cpu: usize) {
        let mut mask = [0u64; 16];
        if cpu < 1024 {
            mask[cpu / 64] = 1 << (cpu % 64);
            unsafe { syscall3(203, 0, size_of_val(&mask), mask.as_ptr() as usize) };
        }
    }

    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    pub fn pin(_: usize) {}

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn affinity() -> Vec<usize> {
        let mut mask = [0u64; 16];
        let r = unsafe { syscall3(204, 0, size_of_val(&mask), mask.as_mut_ptr() as usize) };
        if r <= 0 {
            return Vec::new();
        }
        (0..1024)
            .filter(|&c| mask[c / 64] >> (c % 64) & 1 == 1)
            .collect()
    }

    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    fn affinity() -> Vec<usize> {
        Vec::new()
    }

    /// One entry per logical CPU in the affinity mask: physical cores fastest first, then their siblings.
    pub fn fast_cores() -> Vec<Cpu> {
        let read = |cpu: usize, f: &str| -> i64 {
            std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/{f}"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0)
        };
        let mut cores: Vec<((i64, i64), i64, usize)> = Vec::new();
        let mut spare: Vec<(i64, usize)> = Vec::new();
        let mut top = 0;
        for cpu in affinity() {
            let key = (
                read(cpu, "topology/physical_package_id"),
                read(cpu, "topology/core_id"),
            );
            let freq = read(cpu, "cpufreq/cpuinfo_max_freq");
            top = top.max(freq);
            if cores.iter().any(|c| c.0 == key) {
                spare.push((freq, cpu));
            } else {
                cores.push((key, freq, cpu));
            }
        }
        cores.sort_by_key(|c| c.0);
        cores.sort_by(|a, b| b.1.cmp(&a.1));
        let slow = |f: i64| f * 10 < top * 9;
        let mut out: Vec<Cpu> = cores
            .iter()
            .map(|c| Cpu {
                id: c.2,
                slow: slow(c.1),
                sibling: false,
            })
            .collect();
        out.extend(spare.iter().map(|&(f, id)| Cpu {
            id,
            slow: slow(f),
            sibling: true,
        }));
        out
    }
}
