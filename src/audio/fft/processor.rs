use std::{fmt::Debug, sync::Arc};

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

pub struct FFTProcessor {
    planner: FftPlanner<f32>,
    fft: Arc<dyn Fft<f32>>,
    buffer: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl Debug for FFTProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FFTProcessor")
            .field("fft", &self.fft.len())
            .finish()
    }
}

impl FFTProcessor {
    pub fn new(size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(size);
        Self {
            buffer: vec![Default::default(); fft.len()],
            scratch: vec![Default::default(); fft.get_inplace_scratch_len()],
            fft,
            planner,
        }
    }

    pub fn resize(&mut self, size: usize) {
        self.fft = self.planner.plan_fft_forward(size);
        self.buffer.resize(self.fft.len(), Default::default());
        self.scratch
            .resize(self.fft.get_inplace_scratch_len(), Default::default());
    }

    pub fn process<'a>(&'a mut self, input: impl IntoIterator<Item = f32>) -> &'a [Complex32] {
        for (buffer_elem, input_elem) in self.buffer.iter_mut().zip(
            input
                .into_iter()
                .chain(std::iter::repeat(Default::default())),
        ) {
            *buffer_elem = Complex32::new(input_elem, 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.buffer, &mut self.scratch);
        &self.buffer
    }
}

/// Create a Hann window function for the given number of samples.
pub fn hann_window(n: usize) -> impl Fn(usize) -> f32 {
    move |i: usize| {
        let a0 = 0.5;
        let a1 = 0.5;
        let two_pi_i_over_n = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
        a0 - a1 * two_pi_i_over_n.cos()
    }
}
