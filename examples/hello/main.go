// Command hello shows how to use pdg as a library.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/akhenak/pdg"
)

func main() {
	e, err := pdg.New()
	if err != nil {
		panic(err)
	}
	audio, err := e.Synthesize(context.Background(), "Hello from the pdg library.", pdg.WithSpeed(1))
	if err != nil {
		panic(err)
	}
	f, err := os.Create("hello.wav")
	if err != nil {
		panic(err)
	}
	if err := pdg.WriteWAV(f, audio); err != nil {
		panic(err)
	}
	if err := f.Close(); err != nil {
		panic(err)
	}
	fmt.Printf("hello.wav: %.1f s\n", float64(len(audio))/pdg.SampleRate)
}
