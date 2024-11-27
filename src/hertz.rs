#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct FFTHertz {
    size: usize,
}

impl FFTHertz {
    pub const fn new(fft_size: usize) -> Self {
        Self { size: fft_size }
    }

    pub const fn from_hertz(hertz: usize) -> Self {
        Self {
            size: (hertz - 1) * 2,
        }
    }
    pub const fn from_fft_size(fft_size: usize) -> Self {
        Self { size: fft_size }
    }

    pub const fn fft_size(&self) -> usize {
        self.size
    }
    pub const fn hertz(&self) -> usize {
        self.size / 2 + 1
    }
    /// Round up the size / hertz to the next optimal fft block size.
    pub const fn optimal(&self) -> Self {
        /// Find the next `x` where `x >= minimum` and `x == 2^a * 3^b` where
        /// `a` and `b` are `usize`'s.
        const fn pow2xpow3(minimum: usize) -> usize {
            let mut best = minimum.next_power_of_two();
            let mut p2 = best.trailing_zeros();
            while best > minimum && p2 > 0 {
                p2 -= 1;
                let remain = minimum >> p2;
                let p3 = remain.ilog(3);
                let mut v = (1_usize << p2) * 3_usize.pow(p3);
                if v < minimum {
                    v *= 3;
                }
                if v < best {
                    best = v;
                }
            }
            best
        }

        Self {
            size: pow2xpow3(self.size),
        }
    }
}
