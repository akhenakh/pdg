// Package assets holds the data embedded in the pdn binary.
package assets

import _ "embed"

// Model is the converted Paradee v2g graph (see tools/convert.py).
//
//go:embed paradee.bin
var Model []byte

// Gold is misaki's us_gold lexicon in its front-coded packed form.
//
//go:embed us_gold.fc.txt
var Gold string

// Silver is misaki's us_silver lexicon.
//
//go:embed us_silver.fc.txt
var Silver string

// Config is the model's phoneme vocabulary.
//
//go:embed config.json
var Config []byte
