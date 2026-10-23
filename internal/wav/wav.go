// Package wav writes 16-bit PCM WAV files, including streaming output.
package wav

import (
	"encoding/binary"
	"io"
	"math"
)

// Header returns a 44-byte mono PCM WAV header for dataSize bytes. Pass
// 0xFFFFFFFF for a stream whose length is not known yet.
func Header(sampleRate, dataSize int) []byte {
	h := make([]byte, 44)
	copy(h[0:4], "RIFF")
	binary.LittleEndian.PutUint32(h[4:8], uint32(36+dataSize))
	copy(h[8:12], "WAVE")
	copy(h[12:16], "fmt ")
	binary.LittleEndian.PutUint32(h[16:20], 16)
	binary.LittleEndian.PutUint16(h[20:22], 1) // PCM
	binary.LittleEndian.PutUint16(h[22:24], 1) // mono
	binary.LittleEndian.PutUint32(h[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(h[28:32], uint32(sampleRate*2))
	binary.LittleEndian.PutUint16(h[32:34], 2)
	binary.LittleEndian.PutUint16(h[34:36], 16)
	copy(h[36:40], "data")
	binary.LittleEndian.PutUint32(h[40:44], uint32(dataSize))
	return h
}

// PCM converts float samples to little-endian signed 16-bit bytes.
func PCM(samples []float32) []byte {
	b := make([]byte, len(samples)*2)
	for i, s := range samples {
		if s > 1 {
			s = 1
		} else if s < -1 {
			s = -1
		}
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(math.Round(float64(s)*32767))))
	}
	return b
}

// Write encodes a complete mono 16-bit PCM WAV stream.
func Write(w io.Writer, samples []float32, sampleRate int) error {
	if _, err := w.Write(Header(sampleRate, len(samples)*2)); err != nil {
		return err
	}
	_, err := w.Write(PCM(samples))
	return err
}
