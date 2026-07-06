/// Convert a frequency in Hertz to a linearised value in the range [0, 1].
///
/// This converts the frequency to a logarithmic scale.
/// The table of frequencies is based on the human hearing range, see [linear_frequency_table].
pub fn linearise_frequency(freq: f32) -> f32 {
    let table = linear_frequency_table();
    let freq = freq.ln();

    if freq <= table[0].0 {
        return table[0].1;
    }

    table
        .windows(2)
        .find_map(|w| {
            let (f0, p0) = w[0];
            let (f1, p1) = w[1];

            if freq >= f0 && freq <= f1 {
                let t = (freq - f0) / (f1 - f0);
                Some(p0 + t * (p1 - p0))
            } else {
                None
            }
        })
        .unwrap_or(1.0)
}

macro_rules! count_exprs {
    () => { 0 };
    ($e:expr) => { 1 };
    ($e:expr, $($es:expr),+) => { 1 + count_exprs!($($es),*) };
}

macro_rules! create_linear_frequency_table {
    ($name:ident : $(($freq:expr, $peak:expr)$(,)?$($comment:literal)?)+) => {
        /// Returns a table of logarithmic frequencies and their corresponding linearised values in the range [0, 1].
        ///
        /// The table is based on the human hearing range:
        ///
        /// | Frequency (Hz) | Linearised Value | Description |
        /// |---------------:|-----------------:|:------------|
        $($(#[doc = concat!("|", $freq, "|", $peak, "|", $comment, "|")])?)*
        ///
        pub fn $name() -> [(f32, f32); count_exprs!($($freq),+)] {
            [
                $(($freq, $peak)),+
            ].map(|(f, p): (f32, f32)| (f.ln(), p))
        }
    };
}

create_linear_frequency_table!(linear_frequency_table:
    (20.0, 0.0),     r#"Bottom of human hearing (Sub-bass rumble)"#
    (60.0, 0.15),    r#"Sub-bass floor (Kick drums, heavy bass drops)"#
    (250.0, 0.25),   r#"Upper bass / Warmth (Bass guitar, low synth notes)"#
    (1000.0, 0.55),  r#"Low Midrange (Vocal body, rhythm guitar, piano root notes)"#
    (2000.0, 0.75),  r#"Upper Midrange (Vocal clarity, snare drum snap)"#
    (6000.0, 0.88),  r#"Presence (Guitar bite, cymbal crash accents)"#
    (12000.0, 0.95), r#"Brilliance / Sibilance (High hats, "s" sounds)"#
    (20000.0, 1.0),  r#"The rest of the spectrum up to Nyquist (Air)"#
);
