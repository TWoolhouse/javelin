use std::{any::type_name_of_val, fmt::Debug, time::Duration};

use cpal::{
    StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

use crate::audio::fft::{FFTBufferRX, FFTBufferTX};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    pub inner: cpal::Device,
    pub config: StreamConfig,
    pub name: String,
}

impl Device {
    pub fn new(device: cpal::Device, config: StreamConfig) -> Self {
        Self {
            config,
            name: device
                .description()
                .map_or("Unknown".into(), |desc| desc.name().to_owned()),
            inner: device,
        }
    }
}

impl TryFrom<cpal::Device> for Device {
    type Error = cpal::Error;
    fn try_from(device: cpal::Device) -> Result<Self, Self::Error> {
        let config = device
            .default_output_config()
            .or_else(|_| device.default_input_config())?;
        Ok(Self::new(device, config.into()))
    }
}

impl Device {
    pub fn try_default() -> Result<Self, cpal::Error> {
        // FIXME: This assumes a lot of things about the kind of device / direction
        // Needs to be more explicit
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .or_else(|| host.default_input_device())
            .ok_or_else(|| cpal::Error::from(cpal::ErrorKind::DeviceNotAvailable))?;
        Self::try_from(device)
    }
}

pub struct Stream {
    pub inner: cpal::Stream,
    pub buffer: FFTBufferRX,
}

impl Device {
    pub fn create_stream_as_input_from_output_device(
        &self,
        mut fft_tx: FFTBufferTX,
        fft_rx: FFTBufferRX,
    ) -> Result<Stream, cpal::Error> {
        let channels = self.config.channels as usize;
        let mut scratch = vec![Default::default(); fft_tx.len()];
        let sample_count_input = fft_tx.len() * channels;

        let stream = self.inner.build_input_stream(
            self.config.clone(),
            move |data: &[f32], info| {
                // Remove any extra samples that don't fit into the FFT buffer
                let skip = data.len().saturating_sub(sample_count_input);
                let data_end = &data[skip..];
                let sample_count_output = data_end.len() / channels;

                // Average the channels into mono, write to the scratch
                data_end
                    .chunks(channels)
                    .map(|chunk| chunk.into_iter().sum::<f32>() / channels as f32)
                    .zip(scratch.iter_mut())
                    .for_each(|(s, scratch)| *scratch = s);

                fft_tx.write(&scratch[..sample_count_output]);
            },
            |err| {
                eprintln!("Error occurred on input stream: {}", err);
            },
            Some(Duration::from_secs(5)),
        )?;
        stream.play()?;
        Ok(Stream {
            inner: stream,
            buffer: fft_rx,
        })
    }
}

impl Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(type_name_of_val(self))
            .field("stream", &"active")
            .field("buffer", &self.buffer)
            .finish()
    }
}
