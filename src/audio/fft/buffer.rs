use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct FFTBuffer {
    cursor: usize,
    buffer: Vec<f32>,
    // TODO: Add this feature
    instant_device: cpal::StreamInstant,
    instant_callback: cpal::StreamInstant,
}

#[derive(Debug)]
pub struct FFTBufferTX {
    ring: Arc<Mutex<FFTBuffer>>,
    len: usize,
}

#[derive(Debug)]
pub struct FFTBufferRX {
    ring: Arc<Mutex<FFTBuffer>>,
    buffer: Vec<f32>,
}

unsafe impl Send for FFTBufferTX {}
unsafe impl Send for FFTBufferRX {}

impl FFTBuffer {
    pub fn new(capacity: usize) -> (FFTBufferTX, FFTBufferRX) {
        let buffer = FFTBuffer {
            buffer: vec![f32::default(); capacity],
            cursor: 0,
            instant_device: cpal::StreamInstant::from_nanos(0),
            instant_callback: cpal::StreamInstant::from_nanos(0),
        };

        let ring = Arc::new(Mutex::new(buffer));
        let tx = FFTBufferTX {
            ring: ring.clone(),
            len: capacity,
        };
        let rx = FFTBufferRX {
            ring: ring,
            buffer: vec![f32::default(); capacity],
        };
        (tx, rx)
    }
}

impl FFTBufferTX {
    pub fn write(&mut self, samples: &[f32]) {
        // Get the last [self.len] samples from the input slice
        let samples = if samples.len() > self.len {
            &samples[samples.len() - self.len..]
        } else {
            samples
        };

        let mut ring = self.ring.lock().unwrap();
        let cursor = ring.cursor;
        match samples.split_at_checked(self.len - cursor) {
            // If samples.len() > the amount of space left in the buffer, we need to wrap around and write to the beginning of the buffer
            Some((first, second)) => {
                ring.buffer[cursor..].copy_from_slice(first);
                ring.buffer[..second.len()].copy_from_slice(second);
                ring.cursor = second.len();
            }
            // Otherwise, we can just write to the buffer normally
            None => {
                ring.buffer[cursor..cursor + samples.len()].copy_from_slice(samples);
                ring.cursor += samples.len();
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl FFTBufferRX {
    pub fn slice(&mut self) -> &[f32] {
        {
            let ring = self.ring.lock().unwrap();
            let (first, second) = ring.buffer.split_at(ring.cursor);
            self.buffer[..first.len()].copy_from_slice(first);
            self.buffer[first.len()..].copy_from_slice(second);
        }
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}
