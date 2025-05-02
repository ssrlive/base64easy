#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use base64easy::*;
    use rand::{Rng, SeedableRng};

    // generate random contents of the specified length and test encode/decode roundtrip
    fn roundtrip_random(
        byte_buf: &mut Vec<u8>,
        str_buf: &mut String,
        engine: EngineKind,
        byte_len: usize,
        approx_values_per_byte: u8,
        max_rounds: u64,
    ) {
        // let the short ones be short but don't let it get too crazy large
        let num_rounds = calculate_number_of_rounds(byte_len, approx_values_per_byte, max_rounds);
        let mut r = rand::rngs::SmallRng::from_rng(&mut rand::rng());
        let mut decode_buf = Vec::new();

        for _ in 0..num_rounds {
            byte_buf.clear();
            str_buf.clear();
            decode_buf.clear();
            while byte_buf.len() < byte_len {
                byte_buf.push(r.random::<u8>());
            }

            *str_buf = encode(&byte_buf, engine);
            decode_buf = decode(&str_buf, engine).unwrap();

            assert_eq!(byte_buf, &decode_buf);
        }
    }

    fn calculate_number_of_rounds(byte_len: usize, approx_values_per_byte: u8, max: u64) -> u64 {
        // don't overflow
        let mut prod = approx_values_per_byte as u64;

        for _ in 0..byte_len {
            if prod > max {
                return max;
            }

            prod = prod.saturating_mul(prod);
        }

        prod
    }

    #[test]
    fn roundtrip_random_short_standard() {
        let mut byte_buf: Vec<u8> = Vec::new();
        let mut str_buf = String::new();

        let engine = EngineKind::Standard;
        for input_len in 0..40 {
            roundtrip_random(&mut byte_buf, &mut str_buf, engine, input_len, 4, 10000);
        }
    }

    #[test]
    fn roundtrip_random_with_fast_loop_standard() {
        let mut byte_buf: Vec<u8> = Vec::new();
        let mut str_buf = String::new();

        let engine = EngineKind::Standard;
        for input_len in 40..100 {
            roundtrip_random(&mut byte_buf, &mut str_buf, engine, input_len, 4, 1000);
        }
    }

    #[test]
    fn roundtrip_random_short_no_padding() {
        let mut byte_buf: Vec<u8> = Vec::new();
        let mut str_buf = String::new();

        let engine = EngineKind::StandardNoPad;
        for input_len in 0..40 {
            roundtrip_random(&mut byte_buf, &mut str_buf, engine, input_len, 4, 10000);
        }
    }

    #[test]
    fn roundtrip_random_no_padding() {
        let mut byte_buf: Vec<u8> = Vec::new();
        let mut str_buf = String::new();

        let engine = EngineKind::StandardNoPad;

        for input_len in 40..100 {
            roundtrip_random(&mut byte_buf, &mut str_buf, engine, input_len, 4, 1000);
        }
    }

    #[test]
    fn roundtrip_decode_trailing_10_bytes() {
        // This is a special case because we decode 8 byte blocks of input at a time as much as we can,
        // ideally unrolled to 32 bytes at a time, in stages 1 and 2. Since we also write a u64's worth
        // of bytes (8) to the output, we always write 2 garbage bytes that then will be overwritten by
        // the NEXT block. However, if the next block only contains 2 bytes, it will decode to 1 byte,
        // and therefore be too short to cover up the trailing 2 garbage bytes. Thus, we have stage 3
        // to handle that case.

        for num_quads in 0..25 {
            let mut s: String = "ABCD".repeat(num_quads);
            s.push_str("EFGHIJKLZg");

            let engine = EngineKind::StandardNoPad;
            let decoded = decode(&s, engine).unwrap();
            assert_eq!(num_quads * 3 + 7, decoded.len());

            assert_eq!(s, encode(&decoded, engine));
        }
    }

    #[test]
    fn display_wrapper_matches_normal_encode() {
        let mut bytes = Vec::<u8>::with_capacity(256);

        for i in 0..255 {
            bytes.push(i);
        }
        bytes.push(255);

        let r = encode(&bytes, EngineKind::Standard);
        let str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0BBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6e3x9fn+AgYKDhIWGh4iJiouMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/wMHCw8TFxsfIycrLzM3Oz9DR0tPU1dbX2Nna29zd3t/g4eLj5OXm5+jp6uvs7e7v8PHy8/T19vf4+fr7/P3+/w==";
        assert_eq!(r, str);
    }
}

#[cfg(test)]
mod tests2 {
    use base64easy::*;
    #[test]
    fn encode_engine_slice_error_when_buffer_too_small() {
        for num_triples in 1..100 {
            let input = "AAA".repeat(num_triples);
            let mut vec = vec![0; (num_triples - 1) * 4];
            use std::io::{Error, ErrorKind::InvalidData};
            let err = Error::new(InvalidData, "Output slice too small").to_string();
            let e = EngineKind::Standard;
            assert_eq!(err, encode_slice(&input, &mut vec, e).unwrap_err().to_string());
            vec.push(0);
            assert_eq!(err, encode_slice(&input, &mut vec, e).unwrap_err().to_string());
            vec.push(0);
            assert_eq!(err, encode_slice(&input, &mut vec, e).unwrap_err().to_string());
            vec.push(0);
            assert_eq!(err, encode_slice(&input, &mut vec, e).unwrap_err().to_string());
            vec.push(0);
            assert_eq!(num_triples * 4, encode_slice(&input, &mut vec, e).unwrap());
        }
    }
}
