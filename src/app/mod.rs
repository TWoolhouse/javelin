use std::{collections::VecDeque, time::Instant};

use cpal::StreamInstant;
use tokio::sync::mpsc::UnboundedSender;

pub mod db;
mod history;

use crate::{
    audio::{
        fft::FFTResolution,
        pipeline::{
            PipelineReconfigurable,
            pass::{Pass, Sample},
        },
    },
    host::action::{Action, Actionable},
};

// #[derive(Debug)]
pub struct App {
    pipeline: PipelineReconfigurable,
    pub pass: Pass,

    tui: crate::ui::terminal::State,
}

impl App {
    /// Get the current FFT resolution.
    pub fn resolution(&self) -> &FFTResolution {
        &self.pipeline.capturer.resolution
    }

    /// Resize the FFT buffer to a new resolution, returning the new resolution.
    /// The new resolution may not be the same as the requested resolution, as it will be adjusted to the nearest optimal resolution.
    pub fn resize(&mut self, resolution: FFTResolution) -> FFTResolution {
        let resolution = resolution.optimal();
        self.pipeline.capturer.resolution = resolution;
        self.pipeline.reconfigure();
        resolution
    }

    /// See [Pipeline](Pipeline::run)
    pub fn fft(&mut self) -> Pass {
        self.pipeline.run()
    }
}

impl Default for App {
    fn default() -> Self {
        Self {
            pipeline: PipelineReconfigurable::default(),
            pass: Pass::default(),

            tui: crate::ui::terminal::State::default(),
        }
    }
}

impl Actionable<Action, &UnboundedSender<Action>> for App {
    fn action(&mut self, action: Action, tx: &UnboundedSender<Action>) -> Option<Action> {
        match action {
            Action::Render => {
                self.pass = self.fft();
                None
            }
            Action::FFTSizeUp => {
                self.resize(FFTResolution::from_samples(
                    self.resolution().samples() + 1,
                    self.resolution().sample_rate(),
                ));
                None
            }
            Action::FFTSizeDown => {
                // TODO: Make this go down a step, instead of half, search for the next optimal below.
                // FIXME: Prevent 0 samples
                self.resize(FFTResolution::from_samples(
                    self.resolution().samples() / 2,
                    self.resolution().sample_rate(),
                ));
                None
            }
            _ => Some(action),
        }
    }
}
