import 'dart:convert';
import 'dart:typed_data';
import 'dart:math' as math;

import 'package:tfann/tfann.dart';

/// Simd based column vector type.
class FVector {
  final Float32x4List columnData;
  late final Float32List listView;
  final int nRows;

  /// number of items
  int get length => nRows;

  /// Checks that this vector has only one element, and returns that element.
  ///
  /// Throws a [StateError] if this is empty or has more than one element.
  double get single => listView.single;
  FVector.zero(this.nRows) : columnData = Float32x4List((nRows + 3) ~/ 4) {
    listView = columnData.buffer.asFloat32List(0, nRows);
  }
  FVector.fromList(List<double> list)
      : nRows = list.length,
        columnData = Float32List.fromList(
                list.followedBy([0, 0, 0]).toList(growable: false))
            .buffer
            .asFloat32x4List() {
    listView = columnData.buffer.asFloat32List(0, nRows);
  }
  FVector.fromBuffer(this.nRows, this.columnData) {
    listView = columnData.buffer.asFloat32List(0, nRows);
  }

  factory FVector.generate(int length, double Function(int index) generator) {
    Float32List list = Float32List(roundUp4(length));
    for (int i = 0; i < length; ++i) {
      list[i] = generator(i);
    }
    return FVector.fromBuffer(length, list.buffer.asFloat32x4List());
  }

  /// Builds a new FVector composed from multiple other FVector's
  factory FVector.join(List<FVector> vectors) {
    int nRows = vectors.fold(
        0, (int previousValue, element) => previousValue + element.nRows);
    Float32List newColumnData = Float32List(roundUp4(nRows));
    int currentVec = 0;
    int src = 0;
    for (int dest = 0; dest < nRows; ++dest) {
      while (src == vectors[currentVec].length) {
        src = 0;
        currentVec++;
      }
      newColumnData[dest] = vectors[currentVec][src];
      src++;
    }
    return FVector.fromBuffer(nRows, newColumnData.buffer.asFloat32x4List());
  }
  FVector slice(int offset, int length) {
    FVector newVec = FVector.zero(length);
    assert(offset + length <= nRows && offset >= 0 && length > 0);
    var dest32 = newVec.columnData.buffer.asInt32List();
    var source32 = columnData.buffer.asInt32List(offset * 4);
    int start = 0;
    if (offset % 4 == 0) {
      for (int i = 0; i < length ~/ 4; ++i)
        newVec.columnData[i] = columnData[i + offset ~/ 4];
      start = length - (length % 4);
    }
    for (int i = start; i < length; ++i) {
      dest32[i] = source32[i];
    }
    return newVec;
  }

  // Given a lengths list, splits the vector to consequtive vectors, each defined
  // by a given length.
  List<FVector> sliceByLengths(List<int> lengths, int offset) {
    assert(lengths.fold<int>(
                0, (previousValue, element) => previousValue + element) +
            offset <=
        nRows);
    return List.generate(lengths.length, (index) {
      FVector s = slice(offset, lengths[index]);
      offset += lengths[index];
      return s;
    });
  }

  /// Breaks the vector into [count] different vectors.
  ///
  /// It considers the vector to be a table(LTR,TTB) with [count] columns,
  /// and returns each column as seperate vector.
  /// The length of the vector must be divisable by [count]
  List<FVector> interleavedSplit(int count) {
    assert(nRows % count == 0);
    int eachLength = nRows ~/ count;
    int eachLengthUp4 = roundUp4(eachLength);
    Int32List inputI32 = columnData.buffer.asInt32List();
    return List.generate(count, (index) {
      Int32List output = Int32List(eachLengthUp4);

      for (int p = index, c = 0; p < nRows; p += count, c++) {
        output[c] = inputI32[p];
      }
      return FVector.fromBuffer(eachLength, output.buffer.asFloat32x4List());
    });
  }

  /// Breaks the vector into [count] different vectors.
  ///
  /// It considers the vector to be a table(LTR,TTB) with [count] rows,
  /// and returns each row as seperate vector.
  /// The length of the vector must be divisable by [count]
  List<FVector> split(int count) {
    assert(nRows % count == 0);
    int eachLength = nRows ~/ count;

    return List.generate(
        count, (index) => this.slice(index * eachLength, eachLength));
  }

  /// creates a new vector from selected indices
  FVector cherryPick(List<int> indices) {
    Int32List output = Int32List(roundUp4(indices.length));
    Int32List input = listView.buffer.asInt32List(0, nRows);
    int p = 0;
    for (int index in indices) {
      output[p] = input[index];
      p++;
    }
    return FVector.fromBuffer(indices.length, output.buffer.asFloat32x4List());
  }

  FVector operator *(FVector other) {
    assert(nRows == other.nRows);
    FVector newVec = FVector.zero(nRows);
    for (int i = 0; i < columnData.length; ++i)
      newVec.columnData[i] = columnData[i] * other.columnData[i];
    return newVec;
  }

  /// Checks if two vectors are identical.
  bool equals(FVector other) {
    if (other.nRows != nRows) return false;
    Int32x4 result;
    var side1 = columnData.buffer.asInt32x4List();
    var side2 = other.columnData.buffer.asInt32x4List();
    result = side1[0] ^ side2[0];
    for (int i = 1; i < side1.length; ++i) {
      result |= side1[i] ^ side2[i];
    }
    return ((result.x | result.y | result.z | result.w) == 0);
  }

  /// Multiplies the vector by scalar (immutable)
  FVector scaled(double factor) {
    FVector newVec = FVector.zero(nRows);
    for (int i = 0; i < columnData.length; ++i)
      newVec.columnData[i] = columnData[i].scale(factor);
    return newVec;
  }

  /// Multiplies the vector by scalar (mutable)
  void scale(double factor) {
    for (int i = 0; i < columnData.length; ++i)
      columnData[i] = columnData[i].scale(factor);
  }

  /// concatenate two vectors (immutable)
  FVector concat(FVector other) {
    Float32x4List newColumnData = Float32x4List((nRows + other.nRows + 3) ~/ 4);
    int i;
    for (i = 0; i < columnData.length; ++i) newColumnData[i] = columnData[i];

    if (nRows % 4 == 0) {
      for (int j = 0; j < other.columnData.length; ++j, ++i) {
        newColumnData[i] = other.columnData[j];
      }
    } else {
      if (nRows % 2 == 0) {
        var intView = newColumnData.buffer.asInt64List(nRows * 4);
        var otherSrcView = other.columnData.buffer.asInt64List();
        for (i = 0; i < otherSrcView.length; ++i) intView[i] = otherSrcView[i];
      } else {
        var intView = newColumnData.buffer.asInt32List(nRows * 4);
        var otherSrcView = other.columnData.buffer.asInt32List();
        for (i = 0; i < otherSrcView.length; ++i) intView[i] = otherSrcView[i];
      }
    }
    return FVector.fromBuffer(nRows + other.nRows, newColumnData);
  }

  /// Multiplies each element by itself (immutable)
  FVector squared() {
    FVector newVec = FVector.zero(nRows);
    for (int i = 0; i < columnData.length; ++i)
      newVec.columnData[i] = columnData[i] * columnData[i];
    return newVec;
  }

  /// Calculates the square-root of each element (immutable)
  FVector sqrt() {
    FVector newVec = FVector.zero(nRows);
    for (int i = 0; i < columnData.length; ++i)
      newVec.columnData[i] = columnData[i].sqrt();
    return newVec;
  }

  /// Calculates the absolute value of each element (immutable)
  FVector abs() {
    FVector newVec = FVector.zero(nRows);
    for (int i = 0; i < columnData.length; ++i)
      newVec.columnData[i] = columnData[i].abs();
    return newVec;
  }

  double largestElement() {
    Float32x4 maxQuant = columnData.last.shuffle([
      Float32x4.xyzw,
      Float32x4.xxxx,
      Float32x4.xyyy,
      Float32x4.xyzz
    ][nRows % 4]);
    for (int i = 0; i < columnData.length - 1; ++i)
      maxQuant = maxQuant.max(columnData[i]);
    return math.max(
        math.max(maxQuant.x, maxQuant.y), math.max(maxQuant.z, maxQuant.w));
  }

  double smallestElement() {
    Float32x4 minQuant = columnData.last.shuffle([
      Float32x4.xyzw,
      Float32x4.xxxx,
      Float32x4.xyyy,
      Float32x4.xyzz
    ][nRows % 4]);
    for (int i = 0; i < columnData.length - 1; ++i)
      minQuant = minQuant.min(columnData[i]);
    return math.min(
        math.min(minQuant.x, minQuant.y), math.min(minQuant.z, minQuant.w));
  }

  /// Calculates the sum of all elements.
  double sumElements() {
    Float32x4 sum = columnData[0];
    for (int i = 1; i < columnData.length; ++i) sum += columnData[i];
    return sum.x + sum.y + sum.z + sum.w;
  }

  FVector operator +(FVector other) {
    assert(nRows == other.nRows);
    FVector newVec = FVector.zero(nRows);
    for (int i = 0; i < columnData.length; ++i)
      newVec.columnData[i] = columnData[i] + other.columnData[i];
    return newVec;
  }

  FVector operator -(FVector other) {
    assert(nRows == other.nRows);
    FVector newVec = FVector.zero(nRows);
    for (int i = 0; i < columnData.length; ++i)
      newVec.columnData[i] = columnData[i] - other.columnData[i];
    return newVec;
  }

  void operator []=(int index, double value) {
    assert(index >= 0 && index < nRows);
    listView[index] = value;
  }

  double operator [](int index) {
    assert(index >= 0 && index < nRows);
    return listView[index];
  }

  FVector reversed() {
    FVector newVec = FVector.zero(nRows);
    for (int i = 0; i < nRows; ++i) {
      newVec.listView[i] = this.listView[nRows - 1 - i];
    }
    return newVec;
  }

  /// Applies function `func` to each element (mutable)
  void apply(double Function(double) func,
      [Float32x4 Function(Float32x4)? funcSIMD]) {
    int start = 0;
    if (funcSIMD != null && nRows >= 4) {
      var full4 = nRows ~/ 4;
      for (int i = 0; i < full4; ++i) {
        columnData[i] = funcSIMD(columnData[i]);
      }
      start = full4 * 4;
    }
    if (start < nRows) {
      Float32List input = columnData.buffer.asFloat32List();
      for (int i = start; i < nRows; ++i) {
        input[i] = func(input[i]);
      }
    }
  }

  /// Applies function `func` to each element (immutable)
  FVector applied(double Function(double) func,
      [Float32x4 Function(Float32x4)? funcSIMD]) {
    int start = 0;
    FVector newVec = FVector.zero(nRows);
    if (funcSIMD != null && nRows >= 4) {
      var full4 = nRows ~/ 4;
      for (int i = 0; i < full4; ++i) {
        newVec.columnData[i] = funcSIMD(columnData[i]);
      }
      start = full4 * 4;
    }
    if (start < nRows) {
      Float32List input = columnData.buffer.asFloat32List();
      Float32List result = newVec.columnData.buffer.asFloat32List();
      for (int i = start; i < nRows; ++i) {
        result[i] = func(input[i]);
      }
    }
    return newVec;
  }

  /// Multiplies this vector by the transposed `other` vector.
  ///
  /// Multiplying column vector by row vector creates a matrix.
  FLeftMatrix multiplyTransposed(FVector other) {
    FLeftMatrix result = FLeftMatrix.zero(other.nRows, this.nRows);
    Float32List columnVec = this.columnData.buffer.asFloat32List();
    Float32List rowVec = other.columnData.buffer.asFloat32List();
    for (int r = 0; r < this.nRows; ++r) {
      var resRow = result.rowsData[r].buffer.asFloat32List();
      for (int c = 0; c < other.nRows; ++c) {
        resRow[c] = columnVec[r] * rowVec[c];
      }
    }
    return result;
  }

  /// stores this vector in json format
  Map<String, dynamic> toJson() {
    Map<String, dynamic> json = {};
    json["rows"] = nRows;
    json["data"] =
        columnData.buffer.asFloat32List(0, nRows).toList(growable: false);
    return json;
  }

  FVector.fromJson(Map<String, dynamic> json)
      : nRows = (json["rows"] as int),
        columnData = Float32List.fromList((json["data"] as List)
                .cast<double>()
                .followedBy([0.0, 0.0, 0.0]).toList(growable: false))
            .buffer
            .asFloat32x4List(
                0, ((json["data"] as List<dynamic>).length + 3) ~/ 4);
  List<double> toList() {
    return columnData.buffer.asFloat32List(0, nRows).toList();
  }
}

int roundUp4(int v) {
  return (v + 3) & 0xfffffffffc;
}

/// Simd based arbitrary size matrix
///
/// A matrix which is optimized to be the left operand of matrix multiplication.
class FLeftMatrix {
  List<Float32x4List> rowsData;
  int nColumns, nRows;
  FLeftMatrix.zero(this.nColumns, this.nRows)
      : rowsData =
            List.generate(nRows, (i) => Float32x4List((nColumns + 3) ~/ 4));
  FLeftMatrix.fromList(List<List<double>> lists)
      : nColumns = lists[0].length,
        nRows = lists.length,
        rowsData = lists
            .map((list) => Float32List.fromList(
                    list.followedBy([0, 0, 0]).toList(growable: false))
                .buffer
                .asFloat32x4List(0, (list.length + 3) ~/ 4))
            .toList(growable: false);

  /// multiplies this matrix by a vector, resulting with new vector.
  FVector multiplyVector(FVector vec) {
    assert(vec.nRows == this.nColumns);
    Float32List float32list = Float32List(roundUp4(this.nRows));
    for (int i = 0; i < nRows; ++i) {
      Float32x4List currentRow = rowsData[i];
      Float32x4 sum = Float32x4.zero();
      for (int j = 0; j < currentRow.length; ++j) {
        sum = sum + (currentRow[j] * vec.columnData[j]);
      }
      float32list[i] = sum.x + sum.y + sum.z + sum.w;
    }
    return FVector.fromBuffer(this.nRows, float32list.buffer.asFloat32x4List());
  }

  FLeftMatrix operator *(FRightMatrix right) {
    assert(right.nRows == this.nColumns);
    FLeftMatrix result = FLeftMatrix.zero(right.nColumns, this.nRows);
    for (int r = 0; r < this.nRows; ++r) {
      Float32List resultRow = result.rowsData[r].buffer.asFloat32List();
      for (int c = 0; c < right.nColumns; ++c) {
        Float32x4 sum = Float32x4.zero();
        Float32x4List leftRow = this.rowsData[r];
        Float32x4List rightColumn = right.columnsData[c];
        for (int i = 0; i < leftRow.length; ++i) {
          sum += leftRow[i] * rightColumn[i];
        }
        resultRow[c] = sum.x + sum.y + sum.z + sum.w;
      }
    }
    return result;
  }

  FLeftMatrix operator -(FLeftMatrix right) {
    assert(right.nRows == this.nRows);
    assert(right.nColumns == this.nColumns);
    FLeftMatrix result = FLeftMatrix.zero(this.nColumns, this.nRows);
    for (int r = 0; r < this.nRows; ++r) {
      Float32x4List resultRow = result.rowsData[r].buffer.asFloat32x4List();
      Float32x4List leftRow = this.rowsData[r].buffer.asFloat32x4List();
      Float32x4List rightRow = right.rowsData[r].buffer.asFloat32x4List();
      for (int i = 0; i < resultRow.length; ++i) {
        resultRow[i] = leftRow[i] - rightRow[i];
      }
    }
    return result;
  }

  FLeftMatrix operator +(FLeftMatrix right) {
    assert(right.nRows == this.nRows);
    assert(right.nColumns == this.nColumns);
    FLeftMatrix result = FLeftMatrix.zero(this.nColumns, this.nRows);
    for (int r = 0; r < this.nRows; ++r) {
      Float32x4List resultRow = result.rowsData[r].buffer.asFloat32x4List();
      Float32x4List leftRow = this.rowsData[r].buffer.asFloat32x4List();
      Float32x4List rightRow = right.rowsData[r].buffer.asFloat32x4List();
      for (int i = 0; i < resultRow.length; ++i) {
        resultRow[i] = leftRow[i] + rightRow[i];
      }
    }
    return result;
  }

  FLeftMatrix transposed() {
    // we use int32, because it is faster to convert int32 to int64 and
    // vice-versa, than to convert Float32 to double and vice-versa
    //
    FLeftMatrix result = FLeftMatrix.zero(this.nRows, this.nColumns);
    var int32Views = result.rowsData
        .map((e) => e.buffer.asInt32List())
        .toList(growable: false);
    for (int r = 0; r < nRows; ++r) {
      var currentRow = this.rowsData[r].buffer.asInt32List();
      for (int c = 0; c < nColumns; ++c) {
        int32Views[c][r] = currentRow[c];
      }
    }
    return result;
  }

  FLeftMatrix scaled(double factor) {
    FLeftMatrix newMat = FLeftMatrix.zero(nColumns, nRows);
    for (int i = 0; i < nRows; ++i) {
      Float32x4List currentRow = rowsData[i];
      Float32x4List resultRow = newMat.rowsData[i];

      for (int j = 0; j < currentRow.length; ++j) {
        resultRow[j] = currentRow[j].scale(factor);
      }
    }
    return newMat;
  }

  void scale(double factor) {
    for (int i = 0; i < nRows; ++i) {
      Float32x4List currentRow = rowsData[i];
      for (int j = 0; j < currentRow.length; ++j) {
        currentRow[j] = currentRow[j].scale(factor);
      }
    }
  }

  Map<String, dynamic> toJson() {
    Map<String, dynamic> json = {};
    json["rows"] = nRows;
    json["columns"] = nColumns;
    json["data"] = rowsData
        .map((e) => e.buffer.asFloat32List(0, nColumns).toList(growable: false))
        .toList(growable: false);
    return json;
  }

  FLeftMatrix.fromJson(Map<String, dynamic> json)
      : nRows = (json["rows"] as int),
        nColumns = (json["columns"] as int),
        rowsData = (json["data"] as List<dynamic>)
            .map((list) => Float32List.fromList(list
                    .cast<double>()
                    .followedBy([0.0, 0.0, 0.0]).toList(growable: false))
                .buffer
                .asFloat32x4List(0, (list.length + 3) ~/ 4))
            .toList(growable: false);
}

class FRightMatrix {
  List<Float32x4List> columnsData;
  int nColumns, nRows;
  FRightMatrix.zero(this.nColumns, this.nRows)
      : columnsData =
            List.generate(nColumns, (i) => Float32x4List((nRows + 3) ~/ 4));
  FRightMatrix.fromList(List<List<double>> lists)
      : nColumns = lists.length,
        nRows = lists[0].length,
        columnsData = lists
            .map((list) => Float32List.fromList(
                    list.followedBy([0, 0, 0]).toList(growable: false))
                .buffer
                .asFloat32x4List(0, (list.length + 3) ~/ 4))
            .toList(growable: false);

  Map<String, dynamic> toJson() {
    Map<String, dynamic> json = {};
    json["rows"] = nRows;
    json["columns"] = nColumns;
    json["data"] = columnsData
        .map((e) => e.buffer.asFloat32List(0, nRows).toList(growable: false))
        .toList(growable: false);
    return json;
  }

  FRightMatrix.fromJson(Map<String, dynamic> json)
      : nRows = (json["rows"] as int),
        nColumns = (json["columns"] as int),
        columnsData = (json["data"] as List<dynamic>)
            .map((list) => Float32List.fromList((list as List<dynamic>)
                    .map((e) => e as double)
                    .followedBy([0, 0, 0]).toList(growable: false))
                .buffer
                .asFloat32x4List(0, (list.length + 3) ~/ 4))
            .toList(growable: false);
}
