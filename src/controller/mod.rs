pub trait VolumeController {
    fn volume(&self) -> f32;
    fn volume_set(&self, volume: f32);
}

pub trait IMediaController {
    fn play(&self);
    fn pause(&self);
    fn next(&self);
    fn previous(&self);
}

pub trait DeviceController {
    fn set_as_default(&self);
}

#[cfg(target_os = "windows")]
mod win;

pub fn device(device_id: cpal::DeviceId) -> impl DeviceController + VolumeController {
    // FIXME: propagate errors
    #[cfg(target_os = "windows")]
    {
        return win::WinDeviceController::new(device_id).unwrap();
    }
}
