use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FFTResolution {
    /// Size of the FFT buffer in samples
    size: usize,
    /// Sample rate in Hertz
    sample_rate: usize,
}

impl FFTResolution {
    /// Size of the FFT buffer in samples.
    pub const fn samples(&self) -> usize {
        self.size
    }

    /// Number of samples in the output of the FFT
    pub const fn samples_out(&self) -> usize {
        self.size / 2
    }

    /// Sample rate in Hertz.
    pub const fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Width of each FFT bin in Hertz
    pub const fn hertz(&self) -> f32 {
        self.sample_rate as f32 / self.size as f32
    }

    /// Nyquist frequency in Hertz
    pub const fn nyquist(&self) -> f32 {
        (self.sample_rate / 2) as f32
    }

    /// Duration of the FFT buffer in seconds
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f32(self.size as f32 / self.sample_rate as f32)
    }

    pub const fn from_samples(samples: usize, sample_rate: usize) -> Self {
        Self {
            size: samples,
            sample_rate,
        }
    }

    /// Create a new `FFTResolution` from a desired frequency resolution in Hertz and a sample rate.
    pub const fn from_hertz(hertz: f32, sample_rate: usize) -> Self {
        let size = (sample_rate as f32 / hertz).ceil() as usize;
        Self { size, sample_rate }
    }

    /// Create a new `FFTResolution` from a desired duration and a sample rate.
    pub const fn from_duration(duration: Duration, sample_rate: usize) -> Self {
        Self {
            size: (duration.as_secs_f32() * sample_rate as f32) as usize,
            sample_rate,
        }
    }

    /// Create a new `FFTResolution` with the same ratio of size to sample rate, but with a different sample rate.
    pub const fn with_sample_rate(&self, sample_rate: usize) -> Self {
        if sample_rate == self.sample_rate {
            return *self;
        }

        let ratio = sample_rate as f32 / self.sample_rate as f32;
        Self {
            size: (self.size as f32 * ratio) as usize,
            sample_rate,
        }
    }

    /// Round up the size to the next optimal fft block size.
    pub const fn optimal(&self) -> Self {
        Self {
            size: pow2xpow3(self.size),
            sample_rate: self.sample_rate,
        }
    }
}

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
