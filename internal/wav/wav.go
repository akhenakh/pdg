// Package wav writes 16-bit PCM WAV files.
package wav

import (
	"encoding/binary"
	"io"
	"math"
)

// Write encodes samples as a mono 16-bit PCM WAV stream.
func Write(w io.Writer, samples []float32, sampleRate int) error {
	const bits = 16
	dataSize := len(samples) * 2
	header := make([]byte, 44)
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], uint32(36+dataSize))
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)
	binary.LittleEndian.PutUint16(header[20:22], 1) // PCM
	binary.LittleEndian.PutUint16(header[22:24], 1) // mono
	binary.LittleEndian.PutUint32(header[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(header[28:32], uint32(sampleRate*2))
	binary.LittleEndian.PutUint16(header[32:34], 2)
	binary.LittleEndian.PutUint16(header[34:36], bits)
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], uint32(dataSize))
	if _, err := w.Write(header); err != nil {
		return err
	}
	buf := make([]byte, 8192)
	n := 0
	for _, s := range samples {
		if s > 1 {
			s = 1
		} else if s < -1 {
			s = -1
		}
		v := int16(math.Round(float64(s) * 32767))
		binary.LittleEndian.PutUint16(buf[n:n+2], uint16(v))
		n += 2
		if n == len(buf) {
			if _, err := w.Write(buf); err != nil {
				return err
			}
			n = 0
		}
	}
	if n > 0 {
		_, err := w.Write(buf[:n])
		return err
	}
	return nil
}
