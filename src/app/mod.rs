use std::{collections::VecDeque, fmt::Debug, time::Instant};

use cpal::StreamInstant;

pub mod db;
mod history;

use crate::{
    audio::{
        fft::FFTResolution,
        pipeline::{FFTPass, Pipeline, config::ConfigBuilder},
    },
    host::action::{Action, Actionable, Packet},
};

#[derive(Debug)]
pub struct App {
    pipeline: Pipeline,

    // TODO: Figure out what is needed for history
    pub history_db: VecDeque<(Instant, f32, Vec<f32>)>,
    pub timestamp: StreamInstant,
    pub timestamp_last: StreamInstant,
}

impl App {
    /// Get the current FFT resolution.
    pub fn resolution(&self) -> &FFTResolution {
        &self.pipeline.config().resolution
    }

    /// Resize the FFT buffer to a new resolution, returning the new resolution.
    /// The new resolution may not be the same as the requested resolution, as it will be adjusted to the nearest optimal resolution.
    pub fn resize(&mut self, resolution: FFTResolution) -> FFTResolution {
        let resolution = resolution.optimal();
        self.pipeline.reconfigure(
            self.pipeline
                .config()
                .clone()
                .builder()
                .resolution(resolution),
        );
        resolution
    }

    /// See [Pipeline](Pipeline::stream_fft_pass)
    pub fn fft(&mut self) -> FFTPass {
        self.pipeline.stream_fft_pass()
    }
}

impl Default for App {
    fn default() -> Self {
        Self {
            pipeline: Pipeline::new(ConfigBuilder::new()),
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
