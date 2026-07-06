use std::{mem::MaybeUninit, rc::Rc};

use com_policy_config::{IPolicyConfig, PolicyConfigClient};
use windows::{
    Foundation::TypedEventHandler,
    Media::{
        Control::GlobalSystemMediaTransportControlsSessionManager, MediaPlaybackAutoRepeatMode,
    },
    Win32::{
        Media::Audio::{
            ERole,
            Endpoints::{IAudioEndpointVolume, IAudioEndpointVolumeEx},
            IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator, eMultimedia,
        },
        System::Com::{CLSCTX_ALL, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx},
    },
    core::{HSTRING, Interface, Result},
};

use crate::controller::{DeviceController, VolumeController};

/// A simple RAII wrapper for COM initialization and un-initialization.
#[derive(Debug)]
pub struct Com();

impl Com {
    fn new() -> Result<Self> {
        unsafe {
            CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok()?;
        }
        Ok(Self())
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        unsafe {
            windows::Win32::System::Com::CoUninitialize();
        }
    }
}

/// Controls the volume of a specific audio endpoint device.
#[derive(Debug)]
pub struct WinVol {
    com: Rc<Com>,
    endpoint: IAudioEndpointVolumeEx,
}

impl WinVol {
    pub fn new(com: Rc<Com>, device_id: &cpal::DeviceId) -> Result<Self> {
        unsafe {
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;

            let device = enumerator.GetDevice(&HSTRING::from(device_id.id()))?;

            let volume_api = device
                .Activate::<IAudioEndpointVolumeEx>(CLSCTX_ALL, None)
                .unwrap();

            Ok(Self {
                com,
                endpoint: volume_api,
            })
        }
    }
}

#[derive(Debug)]
struct WinDevicePolicy {
    com: Rc<Com>,
    policy: IPolicyConfig,
}

impl WinDevicePolicy {
    fn new(com: Rc<Com>) -> Result<Self> {
        unsafe {
            let policy_config: IPolicyConfig =
                CoCreateInstance(&PolicyConfigClient, None, CLSCTX_ALL)?;
            Ok(Self {
                com,
                policy: policy_config,
            })
        }
    }
}

#[derive(Debug)]
pub struct WinDeviceController {
    device: cpal::DeviceId,
    vol: WinVol,
    policy: WinDevicePolicy,
}

impl WinDeviceController {
    pub fn new(device: cpal::DeviceId) -> Result<Self> {
        let com = Rc::new(Com::new()?);
        Ok(Self {
            vol: WinVol::new(com.clone(), &device)?,
            policy: WinDevicePolicy::new(com)?,
            device,
        })
    }
}

impl VolumeController for WinDeviceController {
    fn volume(&self) -> f32 {
        unsafe { self.vol.endpoint.GetMasterVolumeLevelScalar().unwrap() }
    }

    fn volume_set(&self, volume: f32) {
        unsafe {
            self.vol
                .endpoint
                .SetMasterVolumeLevelScalar(volume, std::ptr::null())
                .unwrap()
        }
    }
}

impl DeviceController for WinDeviceController {
    fn set_as_default(&self) {
        unsafe {
            self.policy
                .policy
                .SetDefaultEndpoint(&HSTRING::from(self.device.id()), eMultimedia)
                .unwrap()
        }
    }
}

#[derive(Debug)]
pub struct MediaController {
    session_manager: GlobalSystemMediaTransportControlsSessionManager,
}

impl MediaController {
    pub async fn new() -> Result<Self> {
        let session_manager =
            GlobalSystemMediaTransportControlsSessionManager::RequestAsync()?.await?;

        Ok(Self { session_manager })
    }
}

async fn get() -> windows::core::Result<()> {
    GlobalSystemMediaTransportControlsSessionManager::RequestAsync()?
        .await?
        .SessionsChanged(&TypedEventHandler::<
            GlobalSystemMediaTransportControlsSessionManager,
            _,
        >::new(move |sender, _| {
            let mgr = sender.ok()?;
            let sessions = mgr.GetSessions()?;
            let session = mgr.GetCurrentSession()?;

            MediaPlaybackAutoRepeatMode::None
                == session.GetPlaybackInfo()?.AutoRepeatMode()?.Value()?;

            Ok(())
        }))?;

    Ok(())
}
