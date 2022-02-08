//MATRIX LIBRARY

class Matrix {
  constructor(rows, columns) {
    this.rows = rows;
    this.columns = columns;
    this.data = [];
    for (let i = 0; i < rows; i++) {
      this.data[i] = [];
      for (let j = 0; j < columns; j++) {
        this.data[i].push(0);
      }
    }
  }

  setData(array) {
    this.data = array;
  }

  //creates one col matrix
  static fromArray(arr) {
    let newMatrix = new Matrix(arr.length, 1);
    arr.map((v, i) => (newMatrix.data[i] = [v]));
    return newMatrix;
  }

  //creates one row matrix
  static toMatrix(arr, rows, columns) {
    let outputMatrix = new Matrix(rows, columns);
    let ct = 0;
    outputMatrix.data = outputMatrix.data.map((row) =>
      row.map((col) => arr[ct++])
    );
    return outputMatrix;
  }

  static toSquareMatrix(arr) {
    let dimension = Math.sqrt(arr.length);
    let output = new Matrix(dimension, dimension);
    output.data = output.data.map((row, i) =>
      arr.slice(i * dimension, (i + 1) * dimension)
    );
    return output;
  }

  static toArray(matrix) {
    let output = [];
    matrix.data = matrix.data.map((row) => row.map((col) => output.push(col)));
    return output;
  }

  randomize(n) {
    this.data = this.data.map((row) => row.map((col) => Math.random() * 2 - 1));
  }

  add(n) {
    if (n instanceof Matrix) {
      this.data = this.data.map((row, i) =>
        row.map((col, j) => col + n.data[i][j])
      );
    } else this.data = this.data.map((row) => row.map((col) => col + n));
  }

  static subtract(m1, m2) {
    let outputMatrix = new Matrix(m1.rows, m1.columns);
    outputMatrix.data = m1.data.map((row, i) =>
      row.map((col, j) => col - m2.data[i][j])
    );
    return outputMatrix;
  }

  subtract(n) {
    if (n instanceof Matrix) {
      this.data = this.data.map((row, i) =>
        row.map((col, j) => col - n.data[i][j])
      );
    } else this.data = this.data.map((row) => row.map((col) => col + n));
  }

  multiply(n) {
    if (n instanceof Matrix) {
      this.data = this.data.map((row, i) =>
        row.map((col, j) => col * n.data[i][j])
      );
    } else {
      this.data = this.data.map((row) => row.map((col) => col * n));
    }
  }

  static multiply(m1, val) {
    let outputMatrix = new Matrix(m1.rows, m1.columns);
    outputMatrix.data = m1.data.map((row) => row.map((col) => col * val));
    return outputMatrix;
  }

  map(fn) {
    this.data = this.data.map((row) => row.map((col) => fn(col)));
  }

  static map(m, fn) {
    let m2 = new Matrix(m.rows, m.columns);
    m2.data = m2.data.map((row, i) => row.map((col, j) => fn(m.data[i][j])));
    return m2;
  }

  cross(m2) {
    //create new data with m1 rows and m2 cols
    let crossMatrix = new Matrix(this.rows, m2.columns);
    //for each row, find dot product of m1.row1 and m2.col++
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < m2.columns; j++) {
        crossMatrix.data[i][j] = dotProduct(
          this.data[i],
          getColumn(m2.data, j)
        );
      }
    }
    this.data = crossMatrix.data;
  }

  static cross(m1, m2) {
    //create new data with m1 rows and m2 cols
    let crossMatrix = new Matrix(m1.rows, m2.columns);
    //console.log(crossMatrix.print());
    //for each row, find dot product of m1.row1 and m2.col++
    for (let i = 0; i < m1.rows; i++) {
      for (let j = 0; j < m2.columns; j++) {
        crossMatrix.data[i][j] = dotProduct(m1.data[i], getColumn(m2.data, j));
      }
    }
    //console.log(crossMatrix.print());
    return crossMatrix;
  }

  //for polynomial regression
  static inverseForFourByFour(matrix) {
    let data = matrix.data;

    let [
      detArray1,
      detArray2,
      detArray3,
      detArray4,
      detArray5,
      detArray6,
      detArray7,
      detArray8,
      detArray9,
      detArray10,
      detArray11,
      detArray12,
      detArray13,
      detArray14,
      detArray15,
      detArray16,
    ] = getDeterminantArrays(data);

    let determinant =
      data[0][0] * detThree(detArray1) -
      data[0][1] * detThree(detArray2) +
      data[0][2] * detThree(detArray3) -
      data[0][3] * detThree(detArray4);

    let cofactor = [
      [
        detThree(detArray1),
        detThree(detArray2) * -1,
        detThree(detArray3),
        detThree(detArray4) * -1,
      ],
      [
        detThree(detArray5) * -1,
        detThree(detArray6),
        detThree(detArray7) * -1,
        detThree(detArray8),
      ],
      [
        detThree(detArray9),
        detThree(detArray10) * -1,
        detThree(detArray11),
        detThree(detArray12) * -1,
      ],
      [
        detThree(detArray13) * -1,
        detThree(detArray14),
        detThree(detArray15) * -1,
        detThree(detArray16),
      ],
    ];

    matrix.setData(cofactor);
    matrix.transpose();
    matrix.multiply(1 / determinant);
    return matrix;
  }

  //for polynomial regression
  static inverseForThreeByThree(matrix) {
    //Find inverse using the adjugate formula
    //determinants and cofactors
    let data = matrix.data;
    let detArray = [];
    matrix.data.map((row) => row.map((col) => detArray.push(col)));

    let determinant = detThree(detArray);

    let cofactor = [
      [
        detTwo(data[1][1], data[1][2], data[2][1], data[2][2]),
        detTwo(data[1][0], data[1][2], data[2][0], data[2][2]) * -1,
        detTwo(data[1][0], data[1][1], data[2][0], data[2][1]),
      ],
      [
        detTwo(data[0][1], data[0][2], data[2][1], data[2][2]) * -1,
        detTwo(data[0][0], data[0][2], data[2][0], data[2][2]),
        detTwo(data[0][0], data[0][1], data[2][0], data[2][1]) * -1,
      ],
      [
        detTwo(data[0][1], data[0][2], data[1][1], data[1][2]),
        detTwo(data[0][0], data[0][2], data[1][0], data[1][2]) * -1,
        detTwo(data[0][0], data[0][1], data[1][0], data[1][1]),
      ],
    ];

    matrix.setData(cofactor);
    matrix.transpose();
    matrix.multiply(1 / determinant);
    return matrix;
  }

  transpose() {
    let data = this.data;
    this.data = this.data.map((v, i) => getColumn(data, i));
  }

  static transpose(matrix) {
    //make the rows the columns, and vice versa
    let output = new Matrix(matrix.columns, matrix.rows);
    output.data = output.data.map((v, i) => getColumn(matrix.data, i));
    return output;
  }

  //for calculating gradients
  //passed
  static verticalMultiply(array, matrix) {
    matrix.data = matrix.data.map((row, i) =>
      row.map((col, j) => col * array[j])
    );
    return matrix;
  }

  //passed
  static convertToFilterMatrixArray(oneRowMatrix, filterSize) {
    let inputArray = oneRowMatrix.data[0];
    let outputArray = [];
    let outputCount = inputArray.length / filterSize / filterSize;
    for (let i = 0; i < outputCount; i++) {
      let matrix = new Matrix(filterSize, filterSize);
      let inputFields = inputArray.slice(
        i * filterSize * filterSize,
        (i + 1) * filterSize * filterSize
      );
      matrix.data = matrix.data.map((row, i) =>
        row.map((col, j) => inputFields[i * filterSize + j])
      );
      outputArray.push(matrix);
    }
    return outputArray;
  }

  //passed
  static flip(matrix) {
    let output = new Matrix(matrix.rows, matrix.columns);
    output.data = matrix.data.map((row) => reverse(row));
    output.data = reverse(output.data);
    return output;
  }

  static combineH(matrixArray) {
    let outputRows = matrixArray[0].rows;
    let outputCols = matrixArray.reduce((a, c) => a + c.columns, 0);
    let outputMatrix = new Matrix(outputRows, outputCols);

    //for each row in outputMatrix, combine the row of all matrices
    outputMatrix.data.map((row, i) => {
      let newRow = [];
      matrixArray.map((m) => (newRow = newRow.concat(m.data[i])));
      outputMatrix.data[i] = newRow;
    });
    return outputMatrix;
  }

  static combineV(matrixArray) {
    let outputRows = matrixArray.reduce((a, c) => a + c.rows, 0);
    let outputCols = matrixArray[0].columns;
    let outputMatrix = new Matrix(outputRows, outputCols);
    //
    let outputData = [];
    matrixArray.map((m) => {
      outputData = outputData.concat(m.data);
    });
    outputMatrix.data = outputData;
    return outputMatrix;
  }

  //passed
  static addPadding(matrix, layers) {
    let output = new Matrix(
      matrix.rows + layers * 2,
      matrix.columns + layers * 2
    );
    let temp = output.data;

    matrix.data.map((row, i) =>
      row.map((col, j) => {
        temp[i + layers][j + layers] = matrix.data[i][j];
      })
    );

    output.data = temp;
    return output;
  }

  print() {
    console.table(this.data);
  }
}

function reverse(array) {
  let output = [];
  array.map((v, i) => {
    output[i] = array[array.length - i - 1];
  });
  return output;
}

function dotProduct(arr1, arr2) {
  return arr1.reduce((a, c, index) => a + c * arr2[index], 0);
}

function getColumn(data, n) {
  let output = [];
  data.map((row) => output.push(row[n]));
  return output;
}

function detThree(arr) {
  return (
    arr[0] * detTwo(arr[4], arr[5], arr[7], arr[8]) -
    arr[1] * detTwo(arr[3], arr[5], arr[6], arr[8]) +
    arr[2] * detTwo(arr[3], arr[4], arr[6], arr[7])
  );
}

function detTwo(a, b, c, d) {
  return a * d - b * c;
}

//complete
function convolution(inputMatrixArray, filterMatrixArray) {
  //create feature map matrix (f-n+1 dimensions)
  let featureMapArray = []; //array of Matrices
  filterMatrixArray.map((filter) => {
    featureMapArray.push(convolute(inputMatrixArray, filter));
  });
  return featureMapArray;
}

//complete - helper for convolution
function convolute(inputMatrixArray, filter) {
  //apply filter to each inputMatrix, sum all channels
  let rows = inputMatrixArray[0].rows - filter.rows + 1;
  let cols = inputMatrixArray[0].columns - filter.columns + 1;
  let featureMap = new Matrix(rows, cols);
  inputMatrixArray.map((matrix) => {
    featureMap.add(getFeatureMap(matrix, filter));
  });
  return featureMap;
}

//complete - helper for convolute
function getFeatureMap(matrix, filter) {
  let [outputRows, outputCols] = [
    matrix.rows - filter.rows + 1,
    matrix.columns - filter.columns + 1,
  ];
  let output = new Matrix(outputRows, outputCols);
  let temp;
  output.data = output.data.map((row, i) =>
    row.map((col, j) => {
      temp = getSubMatrix(matrix, filter.rows, filter.columns, i, j);
      temp.multiply(filter);
      return Matrix.toArray(temp).reduce((a, c) => a + c, 0);
    })
  );
  return output;
}

//complete - helper for getFeatureMap
function getSubMatrix(matrix, rows, columns, rowIndex, colIndex) {
  let output = new Matrix(rows, columns);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      output.data[i][j] = matrix.data[i + rowIndex][j + colIndex];
    }
  }
  return output;
}

function reluActivation(matrix) {
  matrix.data = matrix.data.map((rows) =>
    rows.map((cols) => (cols < 0 ? 0 : cols))
  );
}

function maxPoolFeatureMap(matrix) {
  let pooledMatrix = new Matrix(matrix.rows / 2, matrix.columns / 2);
  let maxValueIndexArray = [];
  pooledMatrix.data = pooledMatrix.data.map((rows, i) =>
    rows.map((cols, j) => {
      let window = getSubMatrix(matrix, 2, 2, i * 2, j * 2);
      let windowArray = Matrix.toArray(window);
      let maxValue = Math.max(...windowArray);
      maxValueIndexArray.push(windowArray.indexOf(maxValue));
      return maxValue;
    })
  );
  return [pooledMatrix, maxValueIndexArray];
}

function flatten(matrixArray) {
  let outputArray = [];
  matrixArray.map((matrix) => {
    outputArray.push(...Matrix.toArray(matrix));
  });
  return outputArray;
}

(function testImplement() {
  let a = new Matrix(2, 5);
  console.log(a);
});
