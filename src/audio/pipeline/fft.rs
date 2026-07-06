use rodio::math::db_to_linear;

use crate::audio::{
    fft::processor::{HANN_WINDOW_GAIN, hann_window},
    pipeline::{FFTPass, Pipeline, Sample},
};

impl Pipeline {
    /// Run the FFT on the current audio buffer
    /// returning an iterator over the FFT bins in decibels.
    ///
    /// 1. Applies a [Hann window](hann_window)
    /// 2. Run FFT
    /// 3. Calculates the power of each bin
    /// 4. Returns the bin values [-200, 0] dB
    pub fn stream_fft_decibels<'a>(&'a mut self) -> impl ExactSizeIterator<Item = f32> + 'a {
        let audio = self.stream.buffer.slice();
        let hann = hann_window(audio.len());

        let fft_samples = self
            .processor
            .process(audio.into_iter().enumerate().map(move |(i, s)| s * hann(i)));

        let power_factor = {
            let sample_count = self.config.resolution.samples();
            2.0 / (sample_count * sample_count) as f32 / HANN_WINDOW_GAIN
        };
        let min_power = 10.0f32.powf(self.config.db_min as f32 / 10.0);

        fft_samples[..fft_samples.len() / 2]
            .into_iter()
            .map(move |c| {
                let power = c.norm_sqr() * power_factor;
                power.max(min_power).log10() * 10.0
            })
    }

    /// Run the FFT on the current audio buffer and return a [FFTPass] containing the samples and the peak bin index.
    ///
    /// See [stream_fft_decibels](Self::stream_fft_decibels) for details on how the FFT is calculated.
    ///
    /// Each sample is converted to a [`Sample`] struct containing the:
    /// - [frequency](Sample::freq) in Hertz, via [FFTResolution::hertz](crate::audio::fft::FFTResolution::hertz)
    /// - [decibel](Sample::db) in dBFS, from [stream_fft_decibels](Self::stream_fft_decibels)
    /// - [linear amplitude](Sample::amp) in the range [0, 1] via [db_to_linear]
    // /// - [linearised frequency](Sample::freq_lin) in the range [0, 1] via [linearise_frequency]
    pub fn stream_fft_pass(&mut self) -> FFTPass {
        let bin_freq = self.config.resolution.hertz();

        let fft_db = self.stream_fft_decibels();

        // TODO: Add an EQ curve:
        // e.g. the bass are much louder, therefore it's hard to see anything else

        let samples = fft_db
            .enumerate()
            .map(move |(bin, db)| {
                let freq = bin as f32 * bin_freq;
                Sample {
                    freq,
                    db,
                    amp: db_to_linear(db), // TODO: Remove the amp, so the eq can go somewhere else and we don't have to recalc it every time
                }
            })
            .collect::<Vec<_>>();

        let peak = samples
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.amp.total_cmp(&b.amp))
            .map(|(i, _)| i)
            .unwrap_or(0);

        FFTPass { samples, peak }
    }
}
