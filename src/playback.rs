use std::{
    sync::{
        mpsc::{Receiver, Sender},
        Arc,
    },
    time::Duration,
};

use cpal::{traits::*, Device, Sample, StreamConfig};
use rodio::{
    dynamic_mixer::{DynamicMixer, DynamicMixerController},
    Source,
};

use crate::pipeline::FFTPipelineTX;

fn spawn_stream(
    device: Device,
    config: StreamConfig,
    mut mix_rx: DynamicMixer<f32>,
    pipeline_tx: FFTPipelineTX<f32>,
) -> cpal::Stream {
    let stream = device
        .build_output_stream(
            &config.clone(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut pipeline = pipeline_tx.lock();
                for frame in data.chunks_exact_mut(config.channels as usize) {
                    for sample in frame.iter_mut() {
                        *sample = mix_rx.next().unwrap_or(Sample::EQUILIBRIUM);
                    }
                    pipeline.push(frame[0]);
                }
                pipeline.notify(data.len() / config.channels as usize);
            },
            move |err| eprintln!("an error occurred on stream output: {err}"),
            None,
        )
        .unwrap();
    stream.play().unwrap();

    stream
}

macro_rules! decl_control {
    ($name_struct:ident, $name_syncer:ident, $name_events: ident, $(($member:ident, $event:ident, $type:ty, $default:expr)),+) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $name_struct {
            $(
                pub $member: $type,
            )+
        }

        impl Default for $name_struct {
            fn default() -> Self {
                Self {
                    $(
                        $member: $default,
                    )+
                }
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum $name_events {
            $(
                $event($type),
            )+
        }

        #[derive(Debug, Clone)]
        pub struct $name_syncer<'a> {
            control: &'a $name_struct,
        }
        impl <'a> $name_syncer<'a> {
            $(
                pub fn $member(&self) -> $name_events {
                    $name_events::$event(self.control.$member)
                }
            )+
        }
        impl $name_struct {
            pub fn sync(&self) -> $name_syncer {
                $name_syncer {
                    control: self,
                }
            }
        }
    };
}

decl_control!(
    Control,
    ControlSync,
    ControlEvent,
    (play, Play, bool, false),
    (speed, Speed, f32, 1.0),
    (volume, Volume, f32, 1.0)
);

struct Stream {
    #[doc(hidden)]
    /// Keep the stream alive
    _stream: cpal::Stream,
}

pub struct Controller {
    pub control: Control,
    pub events: Sender<ControlEvent>,
}

fn src_into_controllable(control: Control) -> (impl Source, Controller) {
    // TODO: Take this from param.
    let src_raw = rodio::source::SineWave::new(440.0);
    let (tx, rx) = std::sync::mpsc::channel::<ControlEvent>();

    let src = src_raw
        .speed(control.speed)
        .amplify(control.volume)
        .periodic_access(Duration::from_secs_f32(0.5), move |src| {
            for event in rx.try_iter() {
                match event {
                    ControlEvent::Speed(speed) => {
                        src.inner_mut().set_factor(speed);
                    }
                    ControlEvent::Volume(volume) => {
                        src.set_factor(volume);
                    }
                    _ => eprintln!("Unsupported!"),
                }
            }
        });
    let controller = Controller {
        control,
        events: tx,
    };
    (src, controller)
}

fn t() {
    let (src, mut controller) = src_into_controllable(Control::default());
    controller.control.play = false;
    controller
        .events
        .send(controller.control.sync().play())
        .unwrap();
}
