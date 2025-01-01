#![doc = include_str!("../README.md")]

/// The base64 encoding engine kind to use when encoding/decoding data.
#[derive(Debug, Clone, Copy)]
pub enum EngineKind {
    /// Base64 Standard
    Standard,
    /// Base64 StandardNoPad
    StandardNoPad,
    /// Base64 UrlSafe
    UrlSafe,
    /// Base64 UrlSafeNoPad
    UrlSafeNoPad,
}

/// Encode bytes to base64 string.
#[cfg(any(feature = "alloc"))]
pub fn encode<T: AsRef<[u8]>>(bytes: T, engine: EngineKind) -> String {
    match engine {
        EngineKind::Standard => STANDARD.encode(bytes),
        EngineKind::StandardNoPad => STANDARD_NO_PAD.encode(bytes),
        EngineKind::UrlSafe => URL_SAFE.encode(bytes),
        EngineKind::UrlSafeNoPad => URL_SAFE_NO_PAD.encode(bytes),
    }
}

/// Encode bytes to base64 string into a pre-allocated buffer.
pub fn encode_slice<T: AsRef<[u8]>>(
    bytes: T,
    output_buf: &mut [u8],
    engine: EngineKind,
) -> Result<usize, Error> {
    match engine {
        EngineKind::Standard => STANDARD.encode_slice(bytes, output_buf),
        EngineKind::StandardNoPad => STANDARD_NO_PAD.encode_slice(bytes, output_buf),
        EngineKind::UrlSafe => URL_SAFE.encode_slice(bytes, output_buf),
        EngineKind::UrlSafeNoPad => URL_SAFE_NO_PAD.encode_slice(bytes, output_buf),
    }
}

/// Decode base64 string to bytes.
#[cfg(any(feature = "alloc"))]
pub fn decode<T: AsRef<[u8]>>(b64str: T, engine: EngineKind) -> Result<Vec<u8>, Error> {
    match engine {
        EngineKind::Standard => STANDARD.decode(b64str),
        EngineKind::StandardNoPad => STANDARD_NO_PAD.decode(b64str),
        EngineKind::UrlSafe => URL_SAFE.decode(b64str),
        EngineKind::UrlSafeNoPad => URL_SAFE_NO_PAD.decode(b64str),
    }
}

/// Decode base64 string to bytes into a pre-allocated buffer.
pub fn decode_slice<T: AsRef<[u8]>>(
    b64str: T,
    output: &mut [u8],
    engine: EngineKind,
) -> Result<usize, Error> {
    match engine {
        EngineKind::Standard => STANDARD.decode_slice(b64str, output),
        EngineKind::StandardNoPad => STANDARD_NO_PAD.decode_slice(b64str, output),
        EngineKind::UrlSafe => URL_SAFE.decode_slice(b64str, output),
        EngineKind::UrlSafeNoPad => URL_SAFE_NO_PAD.decode_slice(b64str, output),
    }
}

pub(crate) trait Engine: Send + Sync {
    type Config: Config;
    type DecodeEstimate: DecodeEstimate;

    fn config(&self) -> &Self::Config;

    fn internal_encode(&self, input: &[u8], output: &mut [u8]) -> usize;
    fn internal_decoded_len_estimate(&self, input_len: usize) -> Self::DecodeEstimate;
    fn internal_decode(
        &self,
        input: &[u8],
        output: &mut [u8],
        decode_estimate: Self::DecodeEstimate,
    ) -> Result<DecodeMetadata, Error>;

    #[cfg(any(feature = "alloc"))]
    #[inline]
    fn encode<T: AsRef<[u8]>>(&self, input: T) -> String {
        fn inner<E: Engine + ?Sized>(engine: &E, input_bytes: &[u8]) -> String {
            let encoded_size = encoded_len(input_bytes.len(), engine.config().encode_padding())
                .expect("integer overflow when calculating buffer size");

            let mut buf = vec![0; encoded_size];

            encode_with_padding(input_bytes, &mut buf[..], engine, encoded_size);

            String::from_utf8(buf).expect("Invalid UTF8")
        }

        inner(self, input.as_ref())
    }

    #[inline]
    fn encode_slice<T: AsRef<[u8]>>(
        &self,
        input: T,
        output_buf: &mut [u8],
    ) -> Result<usize, Error> {
        fn inner<E>(engine: &E, input_bytes: &[u8], output_buf: &mut [u8]) -> Result<usize, Error>
        where
            E: Engine + ?Sized,
        {
            let encoded_size = encoded_len(input_bytes.len(), engine.config().encode_padding())
                .expect("usize overflow when calculating buffer size");

            if output_buf.len() < encoded_size {
                return Err(Error::OutputSliceTooSmall);
            }

            let b64_output = &mut output_buf[0..encoded_size];

            encode_with_padding(input_bytes, b64_output, engine, encoded_size);

            Ok(encoded_size)
        }

        inner(self, input.as_ref(), output_buf)
    }

    #[cfg(any(feature = "alloc"))]
    #[inline]
    fn decode<T: AsRef<[u8]>>(&self, input: T) -> Result<Vec<u8>, Error> {
        fn inner<E: Engine + ?Sized>(engine: &E, input_bytes: &[u8]) -> Result<Vec<u8>, Error> {
            let estimate = engine.internal_decoded_len_estimate(input_bytes.len());
            let mut buffer = vec![0; estimate.decoded_len_estimate()];

            let bytes_written = engine
                .internal_decode(input_bytes, &mut buffer, estimate)?
                .decoded_len;

            buffer.truncate(bytes_written);

            Ok(buffer)
        }

        inner(self, input.as_ref())
    }

    #[inline]
    fn decode_slice<T: AsRef<[u8]>>(&self, input: T, output: &mut [u8]) -> Result<usize, Error> {
        fn inner<E: Engine + ?Sized>(
            eng: &E,
            input: &[u8],
            output: &mut [u8],
        ) -> Result<usize, Error> {
            eng.internal_decode(
                input,
                output,
                eng.internal_decoded_len_estimate(input.len()),
            )
            .map(|dm| dm.decoded_len)
        }

        inner(self, input.as_ref(), output)
    }
}

#[allow(dead_code)]
pub(crate) trait DecodeEstimate {
    fn decoded_len_estimate(&self) -> usize;
}

#[derive(PartialEq, Eq, Debug)]
pub(crate) struct DecodeMetadata {
    /// Number of decoded bytes output
    pub(crate) decoded_len: usize,
    /// Offset of the first padding byte in the input, if any
    pub(crate) padding_offset: Option<usize>,
}

impl DecodeMetadata {
    pub(crate) fn new(decoded_bytes: usize, padding_index: Option<usize>) -> Self {
        Self {
            decoded_len: decoded_bytes,
            padding_offset: padding_index,
        }
    }
}

/// Calculate the base64 encoded length for a given input length, optionally including any
/// appropriate padding bytes.
///
/// Returns `None` if the encoded length can't be represented in `usize`. This will happen for
/// input lengths in approximately the top quarter of the range of `usize`.
pub const fn encoded_len(bytes_len: usize, padding: bool) -> Option<usize> {
    let rem = bytes_len % 3;

    let complete_input_chunks = bytes_len / 3;
    // `?` is disallowed in const, and `let Some(_) = _ else` requires 1.65.0, whereas this
    // messier syntax works on 1.48
    let complete_chunk_output =
        if let Some(complete_chunk_output) = complete_input_chunks.checked_mul(4) {
            complete_chunk_output
        } else {
            return None;
        };

    if rem > 0 {
        if padding {
            complete_chunk_output.checked_add(4)
        } else {
            let encoded_rem = match rem {
                1 => 2,
                // only other possible remainder is 2
                // can't use a separate _ => unreachable!() in const fns in ancient rust versions
                _ => 3,
            };
            complete_chunk_output.checked_add(encoded_rem)
        }
    } else {
        Some(complete_chunk_output)
    }
}

/// Returns a conservative estimate of the decoded size of `encoded_len` base64 symbols (rounded up
/// to the next group of 3 decoded bytes).
///
/// The resulting length will be a safe choice for the size of a decode buffer, but may have up to
/// 2 trailing bytes that won't end up being needed.
///
/// # Examples
///
/// ```
/// use base64easy::decoded_len_estimate;
///
/// assert_eq!(3, decoded_len_estimate(1));
/// assert_eq!(3, decoded_len_estimate(2));
/// assert_eq!(3, decoded_len_estimate(3));
/// assert_eq!(3, decoded_len_estimate(4));
/// // start of the next quad of encoded symbols
/// assert_eq!(6, decoded_len_estimate(5));
/// ```
pub fn decoded_len_estimate(encoded_len: usize) -> usize {
    STANDARD
        .internal_decoded_len_estimate(encoded_len)
        .decoded_len_estimate()
}

pub(crate) fn encode_with_padding<E: Engine + ?Sized>(
    input: &[u8],
    output: &mut [u8],
    engine: &E,
    expected_encoded_size: usize,
) {
    debug_assert_eq!(expected_encoded_size, output.len());

    let b64_bytes_written = engine.internal_encode(input, output);

    let padding_bytes = if engine.config().encode_padding() {
        add_padding(b64_bytes_written, &mut output[b64_bytes_written..])
    } else {
        0
    };

    let encoded_bytes = b64_bytes_written
        .checked_add(padding_bytes)
        .expect("usize overflow when calculating b64 length");

    debug_assert_eq!(expected_encoded_size, encoded_bytes);
}

pub(crate) const PAD_BYTE: u8 = b'=';

pub(crate) fn add_padding(unpadded_output_len: usize, output: &mut [u8]) -> usize {
    let pad_bytes = (4 - (unpadded_output_len % 4)) % 4;
    // for just a couple bytes, this has better performance than using
    // .fill(), or iterating over mutable refs, which call memset()
    #[allow(clippy::needless_range_loop)]
    for i in 0..pad_bytes {
        output[i] = PAD_BYTE;
    }

    pad_bytes
}

pub(crate) const STANDARD: GeneralPurpose = GeneralPurpose::new(&alphabet::STANDARD, PAD);
pub(crate) const STANDARD_NO_PAD: GeneralPurpose = GeneralPurpose::new(&alphabet::STANDARD, NO_PAD);
pub(crate) const URL_SAFE: GeneralPurpose = GeneralPurpose::new(&alphabet::URL_SAFE, PAD);
pub(crate) const URL_SAFE_NO_PAD: GeneralPurpose = GeneralPurpose::new(&alphabet::URL_SAFE, NO_PAD);

pub(crate) const PAD: GeneralPurposeConfig = GeneralPurposeConfig::new();
pub(crate) const NO_PAD: GeneralPurposeConfig = GeneralPurposeConfig::new()
    .with_encode_padding(false)
    .with_decode_padding_mode(DecodePaddingMode::RequireNone);

#[derive(Debug, Clone)]
pub(crate) struct GeneralPurpose {
    encode_table: [u8; 64],
    decode_table: [u8; 256],
    config: GeneralPurposeConfig,
}

impl Engine for GeneralPurpose {
    type Config = GeneralPurposeConfig;
    type DecodeEstimate = GeneralPurposeEstimate;

    fn internal_encode(&self, input: &[u8], output: &mut [u8]) -> usize {
        let mut input_index: usize = 0;

        const BLOCKS_PER_FAST_LOOP: usize = 4;
        const LOW_SIX_BITS: u64 = 0x3F;

        // we read 8 bytes at a time (u64) but only actually consume 6 of those bytes. Thus, we need
        // 2 trailing bytes to be available to read..
        let last_fast_index = input.len().saturating_sub(BLOCKS_PER_FAST_LOOP * 6 + 2);
        let mut output_index = 0;

        if last_fast_index > 0 {
            while input_index <= last_fast_index {
                // Major performance wins from letting the optimizer do the bounds check once, mostly
                // on the output side
                let input_chunk =
                    &input[input_index..(input_index + (BLOCKS_PER_FAST_LOOP * 6 + 2))];
                let output_chunk =
                    &mut output[output_index..(output_index + BLOCKS_PER_FAST_LOOP * 8)];

                // Hand-unrolling for 32 vs 16 or 8 bytes produces yields performance about equivalent
                // to unsafe pointer code on a Xeon E5-1650v3. 64 byte unrolling was slightly better for
                // large inputs but significantly worse for 50-byte input, unsurprisingly. I suspect
                // that it's a not uncommon use case to encode smallish chunks of data (e.g. a 64-byte
                // SHA-512 digest), so it would be nice if that fit in the unrolled loop at least once.
                // Plus, single-digit percentage performance differences might well be quite different
                // on different hardware.

                let input_u64 = read_u64(&input_chunk[0..]);

                output_chunk[0] = self.encode_table[((input_u64 >> 58) & LOW_SIX_BITS) as usize];
                output_chunk[1] = self.encode_table[((input_u64 >> 52) & LOW_SIX_BITS) as usize];
                output_chunk[2] = self.encode_table[((input_u64 >> 46) & LOW_SIX_BITS) as usize];
                output_chunk[3] = self.encode_table[((input_u64 >> 40) & LOW_SIX_BITS) as usize];
                output_chunk[4] = self.encode_table[((input_u64 >> 34) & LOW_SIX_BITS) as usize];
                output_chunk[5] = self.encode_table[((input_u64 >> 28) & LOW_SIX_BITS) as usize];
                output_chunk[6] = self.encode_table[((input_u64 >> 22) & LOW_SIX_BITS) as usize];
                output_chunk[7] = self.encode_table[((input_u64 >> 16) & LOW_SIX_BITS) as usize];

                let input_u64 = read_u64(&input_chunk[6..]);

                output_chunk[8] = self.encode_table[((input_u64 >> 58) & LOW_SIX_BITS) as usize];
                output_chunk[9] = self.encode_table[((input_u64 >> 52) & LOW_SIX_BITS) as usize];
                output_chunk[10] = self.encode_table[((input_u64 >> 46) & LOW_SIX_BITS) as usize];
                output_chunk[11] = self.encode_table[((input_u64 >> 40) & LOW_SIX_BITS) as usize];
                output_chunk[12] = self.encode_table[((input_u64 >> 34) & LOW_SIX_BITS) as usize];
                output_chunk[13] = self.encode_table[((input_u64 >> 28) & LOW_SIX_BITS) as usize];
                output_chunk[14] = self.encode_table[((input_u64 >> 22) & LOW_SIX_BITS) as usize];
                output_chunk[15] = self.encode_table[((input_u64 >> 16) & LOW_SIX_BITS) as usize];

                let input_u64 = read_u64(&input_chunk[12..]);

                output_chunk[16] = self.encode_table[((input_u64 >> 58) & LOW_SIX_BITS) as usize];
                output_chunk[17] = self.encode_table[((input_u64 >> 52) & LOW_SIX_BITS) as usize];
                output_chunk[18] = self.encode_table[((input_u64 >> 46) & LOW_SIX_BITS) as usize];
                output_chunk[19] = self.encode_table[((input_u64 >> 40) & LOW_SIX_BITS) as usize];
                output_chunk[20] = self.encode_table[((input_u64 >> 34) & LOW_SIX_BITS) as usize];
                output_chunk[21] = self.encode_table[((input_u64 >> 28) & LOW_SIX_BITS) as usize];
                output_chunk[22] = self.encode_table[((input_u64 >> 22) & LOW_SIX_BITS) as usize];
                output_chunk[23] = self.encode_table[((input_u64 >> 16) & LOW_SIX_BITS) as usize];

                let input_u64 = read_u64(&input_chunk[18..]);

                output_chunk[24] = self.encode_table[((input_u64 >> 58) & LOW_SIX_BITS) as usize];
                output_chunk[25] = self.encode_table[((input_u64 >> 52) & LOW_SIX_BITS) as usize];
                output_chunk[26] = self.encode_table[((input_u64 >> 46) & LOW_SIX_BITS) as usize];
                output_chunk[27] = self.encode_table[((input_u64 >> 40) & LOW_SIX_BITS) as usize];
                output_chunk[28] = self.encode_table[((input_u64 >> 34) & LOW_SIX_BITS) as usize];
                output_chunk[29] = self.encode_table[((input_u64 >> 28) & LOW_SIX_BITS) as usize];
                output_chunk[30] = self.encode_table[((input_u64 >> 22) & LOW_SIX_BITS) as usize];
                output_chunk[31] = self.encode_table[((input_u64 >> 16) & LOW_SIX_BITS) as usize];

                output_index += BLOCKS_PER_FAST_LOOP * 8;
                input_index += BLOCKS_PER_FAST_LOOP * 6;
            }
        }

        // Encode what's left after the fast loop.

        const LOW_SIX_BITS_U8: u8 = 0x3F;

        let rem = input.len() % 3;
        let start_of_rem = input.len() - rem;

        // start at the first index not handled by fast loop, which may be 0.

        while input_index < start_of_rem {
            let input_chunk = &input[input_index..(input_index + 3)];
            let output_chunk = &mut output[output_index..(output_index + 4)];

            output_chunk[0] = self.encode_table[(input_chunk[0] >> 2) as usize];
            output_chunk[1] = self.encode_table
                [((input_chunk[0] << 4 | input_chunk[1] >> 4) & LOW_SIX_BITS_U8) as usize];
            output_chunk[2] = self.encode_table
                [((input_chunk[1] << 2 | input_chunk[2] >> 6) & LOW_SIX_BITS_U8) as usize];
            output_chunk[3] = self.encode_table[(input_chunk[2] & LOW_SIX_BITS_U8) as usize];

            input_index += 3;
            output_index += 4;
        }

        if rem == 2 {
            output[output_index] = self.encode_table[(input[start_of_rem] >> 2) as usize];
            output[output_index + 1] =
                self.encode_table[((input[start_of_rem] << 4 | input[start_of_rem + 1] >> 4)
                    & LOW_SIX_BITS_U8) as usize];
            output[output_index + 2] =
                self.encode_table[((input[start_of_rem + 1] << 2) & LOW_SIX_BITS_U8) as usize];
            output_index += 3;
        } else if rem == 1 {
            output[output_index] = self.encode_table[(input[start_of_rem] >> 2) as usize];
            output[output_index + 1] =
                self.encode_table[((input[start_of_rem] << 4) & LOW_SIX_BITS_U8) as usize];
            output_index += 2;
        }

        output_index
    }

    fn internal_decoded_len_estimate(&self, input_len: usize) -> Self::DecodeEstimate {
        GeneralPurposeEstimate::new(input_len)
    }

    fn internal_decode(
        &self,
        input: &[u8],
        output: &mut [u8],
        estimate: Self::DecodeEstimate,
    ) -> Result<DecodeMetadata, Error> {
        decode::decode_helper(
            input,
            estimate,
            output,
            &self.decode_table,
            self.config.decode_allow_trailing_bits,
            self.config.decode_padding_mode,
        )
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[inline]
pub(crate) fn read_u64(s: &[u8]) -> u64 {
    u64::from_be_bytes(s[..8].try_into().unwrap())
}

impl GeneralPurpose {
    pub(crate) const fn new(alphabet: &alphabet::Alphabet, config: GeneralPurposeConfig) -> Self {
        Self {
            encode_table: encode_table(alphabet),
            decode_table: decode_table(alphabet),
            config,
        }
    }
}

pub(crate) const fn encode_table(alphabet: &alphabet::Alphabet) -> [u8; 64] {
    // the encode table is just the alphabet:
    // 6-bit index lookup -> printable byte
    let mut encode_table = [0_u8; 64];
    {
        let mut index = 0;
        while index < 64 {
            encode_table[index] = alphabet.symbols[index];
            index += 1;
        }
    }

    encode_table
}

pub(crate) const INVALID_VALUE: u8 = 255;

pub(crate) const fn decode_table(alphabet: &alphabet::Alphabet) -> [u8; 256] {
    let mut decode_table = [INVALID_VALUE; 256];

    // Since the table is full of `INVALID_VALUE` already, we only need to overwrite
    // the parts that are valid.
    let mut index = 0;
    while index < 64 {
        // The index in the alphabet is the 6-bit value we care about.
        // Since the index is in 0-63, it is safe to cast to u8.
        decode_table[alphabet.symbols[index] as usize] = index as u8;
        index += 1;
    }

    decode_table
}

pub(crate) trait Config {
    fn encode_padding(&self) -> bool;
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GeneralPurposeConfig {
    encode_padding: bool,
    decode_allow_trailing_bits: bool,
    decode_padding_mode: DecodePaddingMode,
}

impl GeneralPurposeConfig {
    pub(crate) const fn new() -> Self {
        Self {
            encode_padding: true,
            decode_allow_trailing_bits: false,
            decode_padding_mode: DecodePaddingMode::RequireCanonical,
        }
    }

    pub(crate) const fn with_encode_padding(self, padding: bool) -> Self {
        Self {
            encode_padding: padding,
            ..self
        }
    }

    pub(crate) const fn with_decode_padding_mode(self, mode: DecodePaddingMode) -> Self {
        Self {
            decode_padding_mode: mode,
            ..self
        }
    }
}

impl Default for GeneralPurposeConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Config for GeneralPurposeConfig {
    fn encode_padding(&self) -> bool {
        self.encode_padding
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodePaddingMode {
    #[allow(dead_code)]
    Indifferent,
    RequireCanonical,
    RequireNone,
}

pub(crate) struct GeneralPurposeEstimate {
    /// input len % 4
    rem: usize,
    #[allow(dead_code)]
    conservative_decoded_len: usize,
}

impl GeneralPurposeEstimate {
    pub(crate) fn new(encoded_len: usize) -> Self {
        let rem = encoded_len % 4;
        Self {
            rem,
            conservative_decoded_len: (encoded_len / 4 + (rem > 0) as usize) * 3,
        }
    }
}

impl DecodeEstimate for GeneralPurposeEstimate {
    fn decoded_len_estimate(&self) -> usize {
        self.conservative_decoded_len
    }
}

pub(crate) mod alphabet {

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct Alphabet {
        pub(crate) symbols: [u8; ALPHABET_SIZE],
    }
    impl Alphabet {
        const fn from_str_unchecked(alphabet: &str) -> Self {
            let mut symbols = [0_u8; ALPHABET_SIZE];
            let source_bytes = alphabet.as_bytes();

            let mut index = 0;
            while index < ALPHABET_SIZE {
                symbols[index] = source_bytes[index];
                index += 1;
            }

            Self { symbols }
        }
    }

    pub(crate) const ALPHABET_SIZE: usize = 64;

    pub(crate) const STANDARD: Alphabet = Alphabet::from_str_unchecked(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
    );
    pub(crate) const URL_SAFE: Alphabet = Alphabet::from_str_unchecked(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
    );
}

pub(crate) mod decode {
    use super::*;

    #[inline]
    pub(crate) fn decode_helper(
        input: &[u8],
        estimate: GeneralPurposeEstimate,
        output: &mut [u8],
        decode_table: &[u8; 256],
        decode_allow_trailing_bits: bool,
        padding_mode: DecodePaddingMode,
    ) -> Result<DecodeMetadata, Error> {
        let input_complete_nonterminal_quads_len =
            complete_quads_len(input, estimate.rem, output.len(), decode_table)?;

        const UNROLLED_INPUT_CHUNK_SIZE: usize = 32;
        const UNROLLED_OUTPUT_CHUNK_SIZE: usize = UNROLLED_INPUT_CHUNK_SIZE / 4 * 3;

        let input_complete_quads_after_unrolled_chunks_len =
            input_complete_nonterminal_quads_len % UNROLLED_INPUT_CHUNK_SIZE;

        let input_unrolled_loop_len =
            input_complete_nonterminal_quads_len - input_complete_quads_after_unrolled_chunks_len;

        // chunks of 32 bytes
        for (chunk_index, chunk) in input[..input_unrolled_loop_len]
            .chunks_exact(UNROLLED_INPUT_CHUNK_SIZE)
            .enumerate()
        {
            let input_index = chunk_index * UNROLLED_INPUT_CHUNK_SIZE;
            let chunk_output = &mut output[chunk_index * UNROLLED_OUTPUT_CHUNK_SIZE
                ..(chunk_index + 1) * UNROLLED_OUTPUT_CHUNK_SIZE];

            decode_chunk_8(
                &chunk[0..8],
                input_index,
                decode_table,
                &mut chunk_output[0..6],
            )?;
            decode_chunk_8(
                &chunk[8..16],
                input_index + 8,
                decode_table,
                &mut chunk_output[6..12],
            )?;
            decode_chunk_8(
                &chunk[16..24],
                input_index + 16,
                decode_table,
                &mut chunk_output[12..18],
            )?;
            decode_chunk_8(
                &chunk[24..32],
                input_index + 24,
                decode_table,
                &mut chunk_output[18..24],
            )?;
        }

        // remaining quads, except for the last possibly partial one, as it may have padding
        let output_unrolled_loop_len = input_unrolled_loop_len / 4 * 3;
        let output_complete_quad_len = input_complete_nonterminal_quads_len / 4 * 3;
        {
            let output_after_unroll =
                &mut output[output_unrolled_loop_len..output_complete_quad_len];

            for (chunk_index, chunk) in input
                [input_unrolled_loop_len..input_complete_nonterminal_quads_len]
                .chunks_exact(4)
                .enumerate()
            {
                let chunk_output = &mut output_after_unroll[chunk_index * 3..chunk_index * 3 + 3];

                decode_chunk_4(
                    chunk,
                    input_unrolled_loop_len + chunk_index * 4,
                    decode_table,
                    chunk_output,
                )?;
            }
        }

        decode_suffix(
            input,
            input_complete_nonterminal_quads_len,
            output,
            output_complete_quad_len,
            decode_table,
            decode_allow_trailing_bits,
            padding_mode,
        )
    }

    pub(crate) fn complete_quads_len(
        input: &[u8],
        input_len_rem: usize,
        output_len: usize,
        decode_table: &[u8; 256],
    ) -> Result<usize, Error> {
        debug_assert!(input.len() % 4 == input_len_rem);

        // detect a trailing invalid byte, like a newline, as a user convenience
        if input_len_rem == 1 {
            let last_byte = input[input.len() - 1];
            // exclude pad bytes; might be part of padding that extends from earlier in the input
            if last_byte != PAD_BYTE && decode_table[usize::from(last_byte)] == INVALID_VALUE {
                return Err(Error::InvalidByte(input.len() - 1, last_byte));
            }
        };

        // skip last quad, even if it's complete, as it may have padding
        let input_complete_nonterminal_quads_len = input
            .len()
            .saturating_sub(input_len_rem)
            // if rem was 0, subtract 4 to avoid padding
            .saturating_sub((input_len_rem == 0) as usize * 4);
        debug_assert!(
            input.is_empty()
                || (1..=4).contains(&(input.len() - input_complete_nonterminal_quads_len))
        );

        // check that everything except the last quad handled by decode_suffix will fit
        if output_len < input_complete_nonterminal_quads_len / 4 * 3 {
            return Err(Error::OutputSliceTooSmall);
        };
        Ok(input_complete_nonterminal_quads_len)
    }

    #[inline(always)]
    pub(crate) fn decode_chunk_8(
        input: &[u8],
        index_at_start_of_input: usize,
        decode_table: &[u8; 256],
        output: &mut [u8],
    ) -> Result<(), Error> {
        let morsel = decode_table[usize::from(input[0])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input, input[0]));
        }
        let mut accum = u64::from(morsel) << 58;

        let morsel = decode_table[usize::from(input[1])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input + 1, input[1]));
        }
        accum |= u64::from(morsel) << 52;

        let morsel = decode_table[usize::from(input[2])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input + 2, input[2]));
        }
        accum |= u64::from(morsel) << 46;

        let morsel = decode_table[usize::from(input[3])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input + 3, input[3]));
        }
        accum |= u64::from(morsel) << 40;

        let morsel = decode_table[usize::from(input[4])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input + 4, input[4]));
        }
        accum |= u64::from(morsel) << 34;

        let morsel = decode_table[usize::from(input[5])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input + 5, input[5]));
        }
        accum |= u64::from(morsel) << 28;

        let morsel = decode_table[usize::from(input[6])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input + 6, input[6]));
        }
        accum |= u64::from(morsel) << 22;

        let morsel = decode_table[usize::from(input[7])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input + 7, input[7]));
        }
        accum |= u64::from(morsel) << 16;

        output[..6].copy_from_slice(&accum.to_be_bytes()[..6]);

        Ok(())
    }

    #[inline(always)]
    pub(crate) fn decode_chunk_4(
        input: &[u8],
        index_at_start_of_input: usize,
        decode_table: &[u8; 256],
        output: &mut [u8],
    ) -> Result<(), Error> {
        let morsel = decode_table[usize::from(input[0])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input, input[0]));
        }
        let mut accum = u32::from(morsel) << 26;

        let morsel = decode_table[usize::from(input[1])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input + 1, input[1]));
        }
        accum |= u32::from(morsel) << 20;

        let morsel = decode_table[usize::from(input[2])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input + 2, input[2]));
        }
        accum |= u32::from(morsel) << 14;

        let morsel = decode_table[usize::from(input[3])];
        if morsel == INVALID_VALUE {
            return Err(Error::InvalidByte(index_at_start_of_input + 3, input[3]));
        }
        accum |= u32::from(morsel) << 8;

        output[..3].copy_from_slice(&accum.to_be_bytes()[..3]);

        Ok(())
    }

    pub(crate) fn decode_suffix(
        input: &[u8],
        input_index: usize,
        output: &mut [u8],
        mut output_index: usize,
        decode_table: &[u8; 256],
        decode_allow_trailing_bits: bool,
        padding_mode: DecodePaddingMode,
    ) -> Result<DecodeMetadata, Error> {
        debug_assert!((input.len() - input_index) <= 4);

        // Decode any leftovers that might not be a complete input chunk of 4 bytes.
        // Use a u32 as a stack-resident 4 byte buffer.
        let mut morsels_in_leftover = 0;
        let mut padding_bytes_count = 0;
        // offset from input_index
        let mut first_padding_offset: usize = 0;
        let mut last_symbol = 0_u8;
        let mut morsels = [0_u8; 4];

        for (leftover_index, &b) in input[input_index..].iter().enumerate() {
            // '=' padding
            if b == PAD_BYTE {
                // There can be bad padding bytes in a few ways:
                // 1 - Padding with non-padding characters after it
                // 2 - Padding after zero or one characters in the current quad (should only
                //     be after 2 or 3 chars)
                // 3 - More than two characters of padding. If 3 or 4 padding chars
                //     are in the same quad, that implies it will be caught by #2.
                //     If it spreads from one quad to another, it will be an invalid byte
                //     in the first quad.
                // 4 - Non-canonical padding -- 1 byte when it should be 2, etc.
                //     Per config, non-canonical but still functional non- or partially-padded base64
                //     may be treated as an error condition.

                if leftover_index < 2 {
                    // Check for error #2.
                    // Either the previous byte was padding, in which case we would have already hit
                    // this case, or it wasn't, in which case this is the first such error.
                    debug_assert!(
                        leftover_index == 0 || (leftover_index == 1 && padding_bytes_count == 0)
                    );
                    let bad_padding_index = input_index + leftover_index;
                    return Err(Error::InvalidByte(bad_padding_index, b));
                }

                if padding_bytes_count == 0 {
                    first_padding_offset = leftover_index;
                }

                padding_bytes_count += 1;
                continue;
            }

            // Check for case #1.
            // To make '=' handling consistent with the main loop, don't allow
            // non-suffix '=' in trailing chunk either. Report error as first
            // erroneous padding.
            if padding_bytes_count > 0 {
                return Err(Error::InvalidByte(
                    input_index + first_padding_offset,
                    PAD_BYTE,
                ));
            }

            last_symbol = b;

            // can use up to 8 * 6 = 48 bits of the u64, if last chunk has no padding.
            // Pack the leftovers from left to right.
            let morsel = decode_table[b as usize];
            if morsel == INVALID_VALUE {
                return Err(Error::InvalidByte(input_index + leftover_index, b));
            }

            morsels[morsels_in_leftover] = morsel;
            morsels_in_leftover += 1;
        }

        // If there was 1 trailing byte, and it was valid, and we got to this point without hitting
        // an invalid byte, now we can report invalid length
        if !input.is_empty() && morsels_in_leftover < 2 {
            return Err(Error::InvalidLength(input_index + morsels_in_leftover));
        }

        match padding_mode {
            DecodePaddingMode::Indifferent => { /* everything we care about was already checked */ }
            DecodePaddingMode::RequireCanonical => {
                // allow empty input
                if (padding_bytes_count + morsels_in_leftover) % 4 != 0 {
                    return Err(Error::InvalidPadding);
                }
            }
            DecodePaddingMode::RequireNone => {
                if padding_bytes_count > 0 {
                    // check at the end to make sure we let the cases of padding that should be InvalidByte get hit
                    return Err(Error::InvalidPadding);
                }
            }
        }

        // When encoding 1 trailing byte (e.g. 0xFF), 2 base64 bytes ("/w") are needed.
        // / is the symbol for 63 (0x3F, bottom 6 bits all set) and w is 48 (0x30, top 2 bits
        // of bottom 6 bits set).
        // When decoding two symbols back to one trailing byte, any final symbol higher than
        // w would still decode to the original byte because we only care about the top two
        // bits in the bottom 6, but would be a non-canonical encoding. So, we calculate a
        // mask based on how many bits are used for just the canonical encoding, and optionally
        // error if any other bits are set. In the example of one encoded byte -> 2 symbols,
        // 2 symbols can technically encode 12 bits, but the last 4 are non-canonical, and
        // useless since there are no more symbols to provide the necessary 4 additional bits
        // to finish the second original byte.

        let leftover_bytes_to_append = morsels_in_leftover * 6 / 8;
        // Put the up to 6 complete bytes as the high bytes.
        // Gain a couple percent speedup from nudging these ORs to use more ILP with a two-way split.
        let mut leftover_num = (u32::from(morsels[0]) << 26)
            | (u32::from(morsels[1]) << 20)
            | (u32::from(morsels[2]) << 14)
            | (u32::from(morsels[3]) << 8);

        // if there are bits set outside the bits we care about, last symbol encodes trailing bits that
        // will not be included in the output
        let mask = !0_u32 >> (leftover_bytes_to_append * 8);
        if !decode_allow_trailing_bits && (leftover_num & mask) != 0 {
            // last morsel is at `morsels_in_leftover` - 1
            return Err(Error::InvalidLastSymbol(
                input_index + morsels_in_leftover - 1,
                last_symbol,
            ));
        }

        // Strangely, this approach benchmarks better than writing bytes one at a time,
        // or copy_from_slice into output.
        for _ in 0..leftover_bytes_to_append {
            let hi_byte = (leftover_num >> 24) as u8;
            leftover_num <<= 8;
            *output
                .get_mut(output_index)
                .ok_or(Error::OutputSliceTooSmall)? = hi_byte;
            output_index += 1;
        }

        let padding_index = if padding_bytes_count > 0 {
            Some(input_index + first_padding_offset)
        } else {
            None
        };
        Ok(DecodeMetadata::new(output_index, padding_index))
    }
}

pub use error::Error;

pub(crate) mod error {
    /// Errors that can occur while encoding/decoding base64.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum Error {
        /// An invalid byte was found in the input. The offset and offending byte are provided.
        ///
        /// Padding characters (`=`) interspersed in the encoded form are invalid, as they may only
        /// be present as the last 0-2 bytes of input.
        ///
        /// This error may also indicate that extraneous trailing input bytes are present, causing
        /// otherwise valid padding to no longer be the last bytes of input.
        InvalidByte(usize, u8),

        /// The length of the input, as measured in valid base64 symbols, is invalid.
        /// There must be 2-4 symbols in the last input quad.
        InvalidLength(usize),

        /// The last non-padding input symbol's encoded 6 bits have nonzero bits that will be discarded.
        /// This is indicative of corrupted or truncated Base64.
        /// Unlike [Error::InvalidByte], which reports symbols that aren't in the alphabet,
        /// this error is for symbols that are in the alphabet but represent nonsensical encodings.
        InvalidLastSymbol(usize, u8),

        /// The nature of the padding was not as configured: absent or incorrect when it must be
        /// canonical, or present when it must be absent, etc.
        InvalidPadding,

        /// The provided slice is too small.
        OutputSliceTooSmall,
    }

    impl core::fmt::Display for Error {
        fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
            match self {
                Self::InvalidByte(index, byte) => {
                    write!(f, "Invalid symbol {}, offset {}.", byte, index)
                }
                Self::InvalidLength(len) => write!(f, "Invalid input length: {}", len),
                Self::InvalidLastSymbol(index, byte) => {
                    write!(f, "Invalid last symbol {}, offset {}.", byte, index)
                }
                Self::InvalidPadding => write!(f, "Invalid padding"),
                Self::OutputSliceTooSmall => write!(f, "Output slice too small"),
            }
        }
    }

    #[cfg(feature = "std")]
    impl std::error::Error for Error {}
}
