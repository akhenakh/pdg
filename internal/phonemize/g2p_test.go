package phonemize

import "testing"

func TestReferencePhonemes(t *testing.T) {
	e, err := New()
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct{ text, want string }{
		{
			"The quick brown fox jumps over the lazy dog near the river bank.",
			"ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ nˌɪɹ ðə ɹˈɪvəɹ bˈæŋk.",
		},
		{
			"In 1990, the city spent $5 million on 21st-century, well-known projects.",
			"ɪn nˌIntˈin nˈIndi, ðə sˈɪTi spˈɛnt fˈIv mˈɪljᵊn dˈɑləɹz ˌɔn twˈɛnti fˈɜɹst sˈɛnʧəɹi, wˌɛlnˈOn pɹˈɑʤˌɛkts.",
		},
		{
			"Dr. Smith said: \"I can't go - it's 1,250 miles away.\"",
			"dˈɑktəɹ smˈɪθ sˈɛd: “ˌI kˈænt ɡˌO — ɪts wˈʌn θˈWzᵊnd tˈu hˈʌndɹəd fˈɪfti mˈIlz əwˈA.”",
		},
	}
	for _, c := range cases {
		got := e.Phonemize(c.text)
		if got != c.want {
			t.Errorf("\ntext: %s\n got: %s\nwant: %s", c.text, got, c.want)
		}
	}
}
