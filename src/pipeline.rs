use std::sync::{Arc, Mutex, MutexGuard};

use cpal::Sample;
use ringbuf::traits::*;

use crate::hertz::FFTHertz;

pub struct FFTPipeline<S: Sample> {
    samples: ringbuf::LocalRb<ringbuf::storage::Heap<S>>,
    count: usize,
}

impl<S: Sample> FFTPipeline<S> {
    pub fn new(capacity: FFTHertz) -> (FFTPipelineTX<S>, FFTPipelineRX<S>) {
        let pipeline = Arc::new(Mutex::new(FFTPipeline {
            samples: ringbuf::LocalRb::new(capacity.fft_size()),
            count: 0,
        }));
        (
            FFTPipelineTX {
                pipeline: Arc::clone(&pipeline),
            },
            FFTPipelineRX { pipeline },
        )
    }

    pub fn push(&mut self, sample: S) {
        self.samples.push_overwrite(sample);
    }
    pub fn recv(&self) -> impl Iterator<Item = &S> {
        self.samples.iter()
    }
    pub fn notify(&mut self, samples: usize) {
        self.count += samples;
    }
}

#[derive(Clone)]
pub struct FFTPipelineTX<S: Sample> {
    pipeline: Arc<Mutex<FFTPipeline<S>>>,
}
impl<S: Sample> FFTPipelineTX<S> {
    pub fn lock(&self) -> FFTPipelineTXGuard<'_, S> {
        FFTPipelineTXGuard {
            pipeline: self.pipeline.lock().unwrap(),
        }
    }
}
#[derive(Clone)]
pub struct FFTPipelineRX<S: Sample> {
    pipeline: Arc<Mutex<FFTPipeline<S>>>,
}
impl<S: Sample> FFTPipelineRX<S> {
    pub fn lock(&self) -> FFTPipelineRXGuard<'_, S> {
        FFTPipelineRXGuard {
            pipeline: self.pipeline.lock().unwrap(),
        }
    }
}

pub struct FFTPipelineTXGuard<'a, S: Sample> {
    pipeline: MutexGuard<'a, FFTPipeline<S>>,
}
impl<S: Sample> FFTPipelineTXGuard<'_, S> {
    pub fn push(&mut self, sample: S) {
        self.pipeline.push(sample);
    }
    pub fn notify(&mut self, samples: usize) {
        self.pipeline.notify(samples)
    }
}

pub struct FFTPipelineRXGuard<'a, S: Sample> {
    pipeline: MutexGuard<'a, FFTPipeline<S>>,
}
impl<S: Sample> FFTPipelineRXGuard<'_, S> {
    pub fn recv(&self) -> impl Iterator<Item = &S> {
        self.pipeline.recv()
    }
}
