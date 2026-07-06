use std::{
    collections::VecDeque,
    fmt::Debug,
    time::{Duration, Instant},
};

use cpal::{StreamInstant, traits::StreamTrait};
use rodio::math::db_to_linear;

mod db;
mod history;

use crate::{
    app::db::linearise_frequency,
    audio::{
        fft::{FFTBuffer, FFTBufferRX, FFTProcessor, FFTResolution, hann_window},
        stream::{Device, create_stream_as_input_from_output_device},
    },
    host::action::{Action, Actionable, Packet},
};

#[derive(Debug)]
pub struct App {
    resolution: FFTResolution,
    processor: FFTProcessor,
    stream: Stream,

    // TODO: Create a `Pipeline` struct that contains the stream, processor, and maybe history for a given configuration.
    // Then it is easier to switch to a new pipeline when a setting changes

    // TODO: Figure out what is needed for history
    pub history_db: VecDeque<(Instant, f32, Vec<f32>)>,
    pub timestamp: StreamInstant,
    pub timestamp_last: StreamInstant,
}

pub struct Stream {
    fft_rx: FFTBufferRX,
    cpal_handle: cpal::Stream,
}

impl Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stream")
            .field("fft_rx", &self.fft_rx.len())
            .field("cpal_stream", &self.cpal_handle.buffer_size())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Sample {
    pub freq: f32,
    pub db: f32,
    pub linear: f32,
    pub freq_lin: f32,
}

#[derive(Debug, Clone)]
pub struct FFTPass {
    pub samples: Vec<Sample>,
    /// Index of the peak sample in the [samples](Self::samples) vector
    pub peak: usize,
}

impl FFTPass {
    pub fn peak_sample(&self) -> &Sample {
        &self.samples[self.peak]
    }
}

const MIN_DB: f32 = -200.0; // TODO: Expose

impl App {
    /// Get the current FFT resolution.
    pub fn resolution(&self) -> &FFTResolution {
        &self.resolution
    }

    /// Resize the FFT buffer to a new resolution, returning the new resolution.
    /// The new resolution may not be the same as the requested resolution, as it will be adjusted to the nearest optimal resolution.
    pub fn resize(&mut self, resolution: FFTResolution) -> FFTResolution {
        // FIXME: Propagate errors
        let device = Device::try_default().expect("No default audio device available");

        let res_new = resolution
            .with_sample_rate(device.config.sample_rate as usize)
            .optimal();
        if res_new == self.resolution {
            return self.resolution;
        }
        self.resolution = res_new;

        self.processor.resize(self.resolution.samples());
        let (fft_tx, fft_rx) = FFTBuffer::new(self.resolution.samples());
        self.stream = Stream {
            fft_rx,
            cpal_handle: create_stream_as_input_from_output_device(device, fft_tx).unwrap(),
        };

        self.resolution
    }

    /// Run the FFT on the current audio buffer
    /// returning an iterator over the FFT bins in decibels.
    ///
    /// 1. Applies a [Hann window](hann_window)
    /// 2. Run FFT
    /// 3. Calculates the power of each bin
    /// 4. Returns the bin values [-200, 0] dB
    pub fn fft_db<'a>(&'a mut self) -> impl ExactSizeIterator<Item = f32> + 'a {
        // self.timestamp = self.stream.fft_rx.instant();
        let audio = self.stream.fft_rx.slice();
        let hann = hann_window(audio.len());

        let fft_samples = self
            .processor
            .process(audio.into_iter().enumerate().map(move |(i, s)| s * hann(i)));

        let power_factor = {
            let sample_count = self.resolution.samples();
            let hann_window_gain = 0.375;
            2.0 / (sample_count * sample_count) as f32 / hann_window_gain
        };
        let min_power = 10.0f32.powf(MIN_DB / 10.0);

        fft_samples[..fft_samples.len() / 2]
            .into_iter()
            .map(move |c| {
                let power = c.norm_sqr() * power_factor;
                power.max(min_power).log10() * 10.0
            })
    }

    /// Run the FFT on the current audio buffer and return a `FFTPass` containing the samples and the peak bin index.
    ///
    /// See [fft_db](Self::fft_db) for details on how the FFT is calculated.
    ///
    /// Each sample is converted to a [`Sample`] struct containing the:
    /// - [frequency](Sample::freq) in Hertz, via [FFTResolution::hertz]
    /// - [decibel value](Sample::db) in dBFS, from [fft_db](Self::fft_db)
    /// - [linear value](Sample::linear) in the range [0, 1] via [db_to_linear]
    /// - [linearised frequency](Sample::freq_lin) in the range [0, 1] via [linearise_frequency]
    pub fn fft(&mut self) -> FFTPass {
        let bin_freq = self.resolution.hertz();

        let fft_db = self.fft_db();

        // TODO: Add an EQ curve:
        // e.g. the bass are much louder, therefore it's hard to see anything else

        let samples = fft_db
            .enumerate()
            .map(move |(bin, db)| {
                let freq = bin as f32 * bin_freq;
                Sample {
                    freq,
                    db,
                    linear: db_to_linear(db),
                    freq_lin: linearise_frequency(freq),
                }
            })
            .collect::<Vec<_>>();

        let peak = samples
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.linear.total_cmp(&b.linear))
            .map(|(i, _)| i)
            .unwrap_or(0);

        FFTPass { samples, peak }
    }
}

impl Default for App {
    fn default() -> Self {
        let device = Device::try_default().expect("No default audio device available");

        let resolution = FFTResolution::from_duration(
            Duration::from_millis(44),
            device.config.sample_rate as usize,
        )
        .optimal();

        let (fft_tx, fft_rx) = FFTBuffer::new(resolution.samples());
        Self {
            resolution,
            processor: FFTProcessor::new(resolution.samples()),
            stream: Stream {
                fft_rx,
                cpal_handle: create_stream_as_input_from_output_device(device, fft_tx).unwrap(),
            },
            history_db: VecDeque::with_capacity(256),
            timestamp: StreamInstant::from_nanos(0),
            timestamp_last: StreamInstant::from_nanos(0),
        }
    }
}

impl Actionable<Packet<Action>> for App {
    fn action(&mut self, packet: Packet<Action>) -> Option<Packet<Action>> {
        // let Packet { action, tx } = packet;
        match packet.action {
            _ => Some(packet),
        }
    }
}
