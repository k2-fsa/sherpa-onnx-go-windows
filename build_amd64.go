//go:build windows && amd64

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/x86_64-pc-windows-gnu -lsherpa-onnx-c-api -lsherpa-onnx-core -lkaldi-native-fbank-core  -lkaldi-decoder-core -lsherpa-onnx-kaldifst-core -lsherpa-onnx-fst -lpiper_phonemize -lespeak-ng -lucd -lonnxruntime
import "C"
