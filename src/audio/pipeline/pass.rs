use rodio::math::{db_to_linear, linear_to_db};
use take_mut::take_or_recover;

#[derive(Debug, Clone)]
pub struct Sample {
    /// Frequency in Hz
    pub freq: f32,
    /// Decibels
    pub db: f32,
}

impl Sample {
    /// [db_to_linear]
    pub fn amp(&self) -> f32 {
        db_to_linear(self.db)
    }

    pub fn with_amp(&self, amp: f32) -> Self {
        Self {
            freq: self.freq,
            db: linear_to_db(amp),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pass {
    pub samples: Vec<Sample>,
    /// Index of the peak sample in the [samples](Self::samples) vector
    pub peak: usize,
}

impl Pass {
    pub fn peak_sample(&self) -> &Sample {
        &self.samples[self.peak]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PassSpec {
    /// Ascending order of frequencies in the [samples](PassBuilder::samples) vector
    pub frequencies: Vec<f32>,
}

impl PassSpec {
    pub fn sample_count(&self) -> usize {
        self.frequencies.len()
    }
}

#[derive(Debug)]
pub struct PassBuilder {
    pub samples: RawSamples,
    /// Index of the peak sample in the [samples](Self::samples) vector
    pub idx_max: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum RawSamples {
    Decibels(Vec<f32>),
    Amplitudes(Vec<f32>),
}

impl RawSamples {
    pub fn as_decibels(&mut self) -> &mut Vec<f32> {
        take_or_recover(
            self,
            || Self::Decibels(vec![1.0, 0.0]),
            |s| match s {
                Self::Amplitudes(amp) => {
                    Self::Decibels(amp.into_iter().map(linear_to_db).collect())
                }
                db => db,
            },
        );
        self.as_inner_mut()
    }
    pub fn as_amplitudes(&mut self) -> &mut Vec<f32> {
        take_or_recover(
            self,
            || Self::Amplitudes(vec![1.0, 0.0]),
            |s| match s {
                Self::Decibels(db) => Self::Amplitudes(db.into_iter().map(db_to_linear).collect()),
                amp => amp,
            },
        );
        self.as_inner_mut()
    }

    pub fn into_decibels(mut self) -> Vec<f32> {
        self.as_decibels();
        self.into_inner()
    }

    pub fn into_amplitudes(mut self) -> Vec<f32> {
        self.as_amplitudes();
        self.into_inner()
    }

    pub fn len(&self) -> usize {
        self.as_inner().len()
    }

    pub fn from_decibels(db: Vec<f32>) -> Self {
        Self::Decibels(db)
    }
    pub fn from_amplitudes(amp: Vec<f32>) -> Self {
        Self::Amplitudes(amp)
    }

    pub fn into_inner(self) -> Vec<f32> {
        match self {
            RawSamples::Decibels(db) => db,
            RawSamples::Amplitudes(amp) => amp,
        }
    }

    pub fn as_inner(&self) -> &Vec<f32> {
        match self {
            RawSamples::Decibels(db) => db,
            RawSamples::Amplitudes(amp) => amp,
        }
    }
    pub fn as_inner_mut(&mut self) -> &mut Vec<f32> {
        match self {
            RawSamples::Decibels(db) => db,
            RawSamples::Amplitudes(amp) => amp,
        }
    }
}

impl PassBuilder {
    fn compute_max(&mut self) -> usize {
        let idx = self
            .samples
            .as_inner()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or_else(|| {
                    if a.is_finite() {
                        std::cmp::Ordering::Greater
                    } else if b.is_finite() {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.idx_max = Some(idx);
        idx
    }

    fn max(&mut self) -> usize {
        self.idx_max.unwrap_or_else(|| self.compute_max())
    }

    pub fn build(mut self, frequencies: &[f32]) -> Pass {
        Pass {
            peak: self.max(),
            samples: self
                .samples
                .into_decibels()
                .into_iter()
                .zip(frequencies)
                .map(|(db, &freq)| Sample { freq, db })
                .collect(),
        }
    }
}
