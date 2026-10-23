// Command hello shows how to use pdgo as a library.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/akhenak/pdgo"
)

func main() {
	e, err := pdgo.New()
	if err != nil {
		panic(err)
	}
	audio, err := e.Synthesize(context.Background(), "Hello from the pdgo library.", pdgo.WithSpeed(1))
	if err != nil {
		panic(err)
	}
	f, err := os.Create("hello.wav")
	if err != nil {
		panic(err)
	}
	if err := pdgo.WriteWAV(f, audio); err != nil {
		panic(err)
	}
	if err := f.Close(); err != nil {
		panic(err)
	}
	fmt.Printf("hello.wav: %.1f s\n", float64(len(audio))/pdgo.SampleRate)
}
