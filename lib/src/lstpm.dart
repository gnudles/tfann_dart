import 'package:tfann/src/activation_function.dart';
import 'package:tfann/src/linalg.dart';
import 'package:tfann/src/network.dart';

class PersistentMemoryTrack {
  double learningEase;
  double forgetingEase;
  int width;
  PersistentMemoryTrack(this.width, this.learningEase, this.forgetingEase);
}

class LSTPMState {
  LSTPMState(this.persistentMemory, this.outputWithMetadata);
  FVector persistentMemory;
  FVector outputWithMetadata;
}

class LSTPM {
  LSTPM(this.tracks, this.forget, this.update, this.processor, this.terminal);
  factory LSTPM.create(List<PersistentMemoryTrack> tracks, int inputWidth,
      int outputWidth, int metadataWidth,
      {int forgetHiddenNeurons = 30,
      int updateHiddenNeurons = 30,
      processorHiddenNeurons = 60,
      terminalHidden1 = 60,
      terminalHidden2 = 60}) {
    int totalTracksWidth = tracks.fold<int>(
        0, (previousValue, element) => previousValue + element.width);
    TfannNetwork forget = TfannNetwork.full([
      inputWidth + outputWidth + metadataWidth,
      forgetHiddenNeurons,
      totalTracksWidth
    ], [
      ActivationFunctionType.uscsls,
      ActivationFunctionType.logistic
    ]);
    TfannNetwork update = TfannNetwork.full([
      inputWidth + outputWidth + metadataWidth,
      updateHiddenNeurons,
      totalTracksWidth
    ], [
      ActivationFunctionType.uscsls,
      ActivationFunctionType.logistic
    ]);
    TfannNetwork processor = TfannNetwork.full([
      inputWidth + outputWidth + metadataWidth,
      processorHiddenNeurons,
      totalTracksWidth
    ], [
      ActivationFunctionType.uscsls,
      ActivationFunctionType.logistic
    ]);
    TfannNetwork terminal = TfannNetwork.full([
      inputWidth + totalTracksWidth,
      terminalHidden1,
      terminalHidden2,
      outputWidth + metadataWidth
    ], [
      ActivationFunctionType.uscsls,
      ActivationFunctionType.funnyHat,
      ActivationFunctionType.cubicSigmoid
    ]);
    return LSTPM(tracks, forget, update, processor, terminal);
  }

  List<PersistentMemoryTrack> tracks;
  TfannNetwork forget;
  TfannNetwork update;
  TfannNetwork processor;
  TfannNetwork terminal;
}
