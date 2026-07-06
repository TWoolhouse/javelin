use std::time::Duration;

use cpal::{
    StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

use crate::audio::fft::FFTBufferTX;

#[derive(Debug)]
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
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .or_else(|| host.default_input_device())
            .ok_or_else(|| cpal::Error::from(cpal::ErrorKind::DeviceNotAvailable))?;
        Self::try_from(device)
    }
}

pub fn create_stream_as_input_from_output_device(
    device: Device,
    mut fft_tx: FFTBufferTX,
) -> Result<cpal::Stream, cpal::Error> {
    let channels = device.config.channels as usize;
    let mut scratch = vec![Default::default(); fft_tx.len()];

    let stream = device.inner.build_input_stream(
        device.config.clone(),
        move |data: &[f32], info| {
            // Remove any extra samples that don't fit into the FFT buffer
            let data = &data[data.len().saturating_sub(fft_tx.len()) * channels..];
            let sample_count = data.len() / channels;

            // Average the channels into mono, write to the scratch
            data.chunks(channels)
                .map(|chunk| chunk.into_iter().sum::<f32>() / channels as f32)
                .zip(scratch.iter_mut())
                .for_each(|(s, scratch)| *scratch = s);

            fft_tx.write(&scratch[..sample_count]);
        },
        |err| {
            eprintln!("Error occurred on input stream: {}", err);
        },
        Some(Duration::from_secs(5)),
    )?;
    stream.play()?;
    Ok(stream)
}
