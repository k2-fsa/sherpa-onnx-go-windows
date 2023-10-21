//go:build windows && 386

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/i686-pc-windows-gnu -lsherpa-onnx-c-api -lsherpa-onnx-core -lkaldi-native-fbank-core -lonnxruntime -lkaldi-decoder-core -lsherpa-onnx-kaldifst-core -lsherpa-onnx-fst
import "C"
