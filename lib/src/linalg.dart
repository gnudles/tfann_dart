import 'dart:typed_data';

class FVector {
  Float32x4List columnData;
  int nRows;
  FVector(this.nRows) : columnData = Float32x4List((nRows + 3) ~/ 4);
  FVector.fromList(List<double> list)
      : nRows = list.length,
        columnData = Float32List.fromList(
                list.followedBy([0, 0, 0]).toList(growable: false))
            .buffer
            .asFloat32x4List();
  FVector.fromBuffer(this.nRows, this.columnData);
  FVector operator *(FVector other) {
    assert(nRows == other.nRows);
    FVector newVec = FVector(nRows);
    for (int i = 0; i < columnData.length; ++i)
      newVec.columnData[i] = columnData[i] * other.columnData[i];
    return newVec;
  }

  apply(double Function(double) func) {
    Float32List l = columnData.buffer.asFloat32List();
    for (int i = 0; i < nRows; ++i) {
      l[i] = func(l[i]);
    }
  }

  Map<String, dynamic> toJson() {
    Map<String, dynamic> json = {};
    json["rows"] = nRows;
    json["data"] =
        columnData.buffer.asFloat32List(0, nRows).toList(growable: false);
    return json;
  }
}

int roundUp4(int v) {
  return (v + 3) & 0xfffffffffc;
}

class FLeftMatrix {
  List<Float32x4List> rowsData;
  int nColumns, nRows;
  FLeftMatrix(this.nColumns, this.nRows)
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
  FVector multiplyVector(FVector vec) {
    assert(vec.nRows == this.nColumns);
    Float32List float32list = Float32List(roundUp4(this.nRows));
    for (int i = 0; i < nRows; ++i) {
      Float32x4List currentRow = rowsData[i];
      Float32x4 sum = Float32x4.zero();
      for (int j = 0; j < currentRow.length; ++j) {
        sum = sum + (currentRow[j] * vec.columnData[j]);
      }
      float32list[i] = sum.w + sum.x + sum.y + sum.z;
    }
    return FVector.fromBuffer(this.nRows, float32list.buffer.asFloat32x4List());
  }

  FLeftMatrix operator *(FRightMatrix right) {
    assert(right.nRows == this.nColumns);
    FLeftMatrix result = FLeftMatrix(this.nRows, right.nColumns);
    for (int r = 0; r < this.nRows; ++r) {
      Float32List resultRow = result.rowsData[r].buffer.asFloat32List();
      for (int c = 0; c < right.nColumns; ++c) {
        Float32x4 sum = Float32x4.zero();
        Float32x4List leftRow = this.rowsData[r];
        Float32x4List rightColumn = right.columnsData[c];
        for (int i = 0; i < leftRow.length; ++i) {
          sum += leftRow[i] * rightColumn[i];
        }
        resultRow[c] = sum.w + sum.x + sum.y + sum.z;
      }
    }
    return result;
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
}

class FRightMatrix {
  List<Float32x4List> columnsData;
  int nColumns, nRows;
  FRightMatrix(this.nColumns, this.nRows)
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
}
