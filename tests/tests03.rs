#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {

    use base64easy::*;

    fn test_encode_decode<T: AsRef<[u8]>>(raw_data: T, encoded_data: &str, decoded_data: T, engine: EngineKind) {
        let encoded = encode(raw_data, engine);
        assert_eq!(encoded_data, encoded);
        let decoded = decode(encoded_data, engine).unwrap();
        assert_eq!(decoded_data.as_ref(), decoded.as_slice());
    }

    #[test]
    fn test_encode_all() {
        let ascii: Vec<u8> = (0..=127).collect();
        let encoded_data = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0\
        BBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6e3x9fn8=";
        test_encode_decode(&ascii, encoded_data, &ascii, EngineKind::Standard);

        let src = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let encoded_data = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODkrLw==";
        test_encode_decode(src, encoded_data, src, EngineKind::Standard);

        let src = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let encoded_data = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODkrLw";
        test_encode_decode(src, encoded_data, src, EngineKind::StandardNoPad);

        let src: Vec<u8> = (0..=63).collect();
        let encoded_data = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0-Pw==";
        test_encode_decode(&src, encoded_data, &src, EngineKind::UrlSafe);

        let src: Vec<u8> = (64..=127).collect();
        let encoded_data = "QEFCQ0RFRkdISUpLTE1OT1BRUlNUVVZXWFlaW1xdXl9gYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXp7fH1-fw";
        test_encode_decode(&src, encoded_data, &src, EngineKind::UrlSafeNoPad);
    }
}

#[cfg(test)]
mod tests2 {
    use base64easy::*;

    fn test_encode_decode<T: AsRef<[u8]>>(raw_data: T, encoded_data: &str, engine: EngineKind) {
        let encoded_len0 = encoded_len(raw_data.as_ref().len(), engine.padding()).unwrap();
        let mut output_buf = vec![0u8; encoded_len0];

        let encoded_len = encode_slice(raw_data.as_ref(), &mut output_buf, engine).unwrap();
        assert_eq!(encoded_data.as_bytes(), &output_buf[..encoded_len]);

        let decoded_len0 = decoded_len_estimate(encoded_len);
        let mut decoded_buf = vec![0u8; decoded_len0];
        let decoded_len = decode_slice(encoded_data, &mut decoded_buf, engine).unwrap();
        assert_eq!(raw_data.as_ref(), &decoded_buf[..decoded_len]);
    }

    #[test]
    fn test_all() {
        let ascii: Vec<u8> = (0..=127).collect();
        let encoded_data = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0\
        BBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6e3x9fn8=";
        test_encode_decode(&ascii, encoded_data, EngineKind::Standard);

        let src = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let encoded_data = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODkrLw==";
        test_encode_decode(src, encoded_data, EngineKind::Standard);

        let src = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let encoded_data = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODkrLw";
        test_encode_decode(src, encoded_data, EngineKind::StandardNoPad);

        let src: Vec<u8> = (0..=63).collect();
        let encoded_data = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0-Pw==";
        test_encode_decode(&src, encoded_data, EngineKind::UrlSafe);

        let src: Vec<u8> = (64..=127).collect();
        let encoded_data = "QEFCQ0RFRkdISUpLTE1OT1BRUlNUVVZXWFlaW1xdXl9gYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXp7fH1-fw";
        test_encode_decode(&src, encoded_data, EngineKind::UrlSafeNoPad);
    }
}
