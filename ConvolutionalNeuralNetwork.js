//Performs convolution with any number of inputs and filters
//Returns feature maps
function convolution(inputMatrixArray, filterMatrixArray) {
  let featureMapArray = []; //array of Matrices
  filterMatrixArray.map((filter) => {
    featureMapArray.push(convolute(inputMatrixArray, filter));
  });
  return featureMapArray;

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
    return matrix;
  }
}

//Performs max-pooling on a matrix
//Returns pooled feature maps
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

//Flattens a matrix array to one column
function flatten(matrixArray) {
  let outputArray = [];
  matrixArray.map((matrix) => {
    outputArray.push(...Matrix.toArray(matrix));
  });
  return outputArray;
}

//Takes in array of filter Matrices and positions of the pooling, returns array of matrices
//returns a matrix twice the size with the same values populated at the Index of the max Values
//passed
function reversePool(matrix, maxValueIndexArray) {
  //for all terms in matrix, get the
  let flatMatrix = Matrix.toArray(matrix);
  let forCombine = []; //ends up with matrix.length count of 2x2 matrices
  flatMatrix.map((val, i) => {
    let array = [0, 0, 0, 0];
    array[maxValueIndexArray[i]] = val;
    forCombine.push(Matrix.toMatrix(array, 2, 2));
  });

  let perRow = Math.sqrt(forCombine.length);
  let rowArray = [];

  for (let i = 0; i < forCombine.length / perRow; i++) {
    let temp = Matrix.combineH(forCombine.slice(i * perRow, (i + 1) * perRow));
    rowArray.push(temp);
  }
  return Matrix.combineV(rowArray);
}

//takes in two matrices, returns matrix
//Example parameters for LeNet-5 architecture: 32, 6, 5, 16, 5, 120, 60, 5

class ConvolutionalNeuralNetwork {
  /* takes in array:
  [dimension of input image, 
    num of filters (6), filter size (5), 
    num of filters (16), filter size (5), 
    ... hidden layers (120, 84), output nodes (10)] */

  constructor(array) {
    //Set filters
    this.learning_rate = 0.6;
    this.inputImageSize = array[0];
    this.filterMatrixArray1 = []; //F1 filters
    this.filterMatrixArray2 = []; //F2 filters
    this.filter1Size = array[2]; //6
    this.filter2Size = array[4]; //16
    let mlpInputCount =
      Math.pow(((array[0] - array[2] + 1) / 2 - array[4] + 1) / 2, 2) *
      array[3]; //length of A: 400
    this.mlp = new MultiLayerPerceptron([mlpInputCount, ...array.slice(5)]); //initialize mlp

    //initialize weights for F1 (6 filters)
    for (let i = 0; i < array[1]; i++) {
      let filterMatrix = new Matrix(array[2], array[2]);
      filterMatrix.randomize();
      this.filterMatrixArray1.push(filterMatrix);
    }

    //initialize weights for F2 (16 filters)
    for (let j = 0; j < array[3]; j++) {
      let filterMatrix = new Matrix(array[4], array[4]);
      filterMatrix.randomize();
      this.filterMatrixArray2.push(filterMatrix);
    }
  }

  //feed image through conv layers
  feedForward(inputImageMatrix) {
    //first convolution layer (F1 + relu + pooling)
    let output = convolution([inputImageMatrix], this.filterMatrixArray1); //conv
    output = output.map((featureMap) => reluActivation(featureMap)); //relu
    output = output.map((featureMap) => maxPoolFeatureMap(featureMap)[0]); //pooling

    //second convolution layer (F2 + relu + pooling)
    output = convolution(output, this.filterMatrixArray2); //conv
    output = output.map((featureMap) => reluActivation(featureMap)); //relu
    output = output.map((featureMap) => maxPoolFeatureMap(featureMap)[0]); //pooling

    let flattenedFeatureMaps = flatten(output); //flatten into 400 length array
    return this.mlp.predict(flattenedFeatureMaps); //feed array through mlp, get output O
  }

  train(inputImageMatrix, targetArray) {
    let convolutionOutputs = []; // store [C1, C2]
    let poolingOutputs = []; //store [P1,P2]
    let maxPoolIndexArrayLayer1 = []; //store index of max values in first pooling
    let maxPoolIndexArrayLayer2 = []; //stpre index of max values in second pooling

    //first convolution layer
    let output = convolution([inputImageMatrix], this.filterMatrixArray1); //(1 image, 6 filters)
    convolutionOutputs.push[output]; // store C1 28x28x6 NOTE: move to after relu if not working
    output = output.map((featureMap) => reluActivation(featureMap));
    output = output.map((featureMap) => {
      maxPoolIndexArrayLayer1.push(maxPoolFeatureMap(featureMap)[1]); //add max value index to array
      return maxPoolFeatureMap(featureMap)[0];
    }); //P1
    poolingOutputs.push(output); // store P1 14x14x6

    //second convolution layer
    output = convolution(output, this.filterMatrixArray2);
    convolutionOutputs.push[output]; //store C2 10x10x16
    output = output.map((featureMap) => reluActivation(featureMap));
    output = output.map((featureMap) => {
      maxPoolIndexArrayLayer2.push(maxPoolFeatureMap(featureMap)[1]);
      return maxPoolFeatureMap(featureMap)[0];
    }); //P2
    poolingOutputs.push(output); // store P2 5x5x16

    let flattenedFeatureMaps = flatten(output); //A1 (400 array)
    let outputArray = this.mlp.predict(flattenedFeatureMaps); //O (10 array)

    //STEP 1 get delA
    //delE - delP matrices
    let delA_1 = new Matrix(1, outputArray.length);
    delA_1.data = [
      // outputArray.map((o, i) => (o - targetArray[i]) * o * (1 - o)),
      outputArray.map((o, i) => (o > 0 ? o - targetArray[i] : 0)),
    ]; //term 1 of delA

    //get other components of delA
    let weightsArray = this.mlp.weights_array; //u, w, v
    let outputsArray = this.mlp.outputs_array; //I, H, O -> one col matrix

    let delOutputsArray = outputsArray.map((m) => Matrix.toArray(m)); //convert to 1 row arrays
    delOutputsArray = delOutputsArray.map((array) =>
      array.map((val) => val * (1 - val))
    ); //convert to delsigmoid i, h, o

    let delA_2 = weightsArray[2];
    let delA_3 = weightsArray[1];

    let delA_4 = weightsArray[0];

    //Final output for delA - one row matrix with 400 cols
    let delA = Matrix.cross(
      Matrix.cross(Matrix.cross(delA_1, delA_2), delA_3),
      delA_4
    );

    //STEP 2
    //convert to array of filter Matrices -> 16 matrices of 5x5
    let delAMatrixArray = Matrix.convertToFilterMatrixArray(
      delA,
      this.filter2Size
    );

    //STEP 3 get delAPrime - reverse pooling for each -> 16 matrices of 10x10
    let delAPrime = delAMatrixArray.map((m, i) =>
      reversePool(m, maxPoolIndexArrayLayer2[i])
    );

    //STEP 4 get delF2
    let p1 = poolingOutputs[0]; //array of 5x5 matrices
    let delF2 = convolution(p1, delAPrime);

    //STEP 5
    //adjust F2 after Step 6

    //STEP 6 get delP1 -> used to adjust F1 (FULL Convolution between delAPrime and inverted F)
    let invertedFilterMatrixArray2 = this.filterMatrixArray2.map((m) =>
      Matrix.flip(m)
    );

    let delP1 = [];
    delAPrime.map((m, i) => {
      delP1.push(
        getFeatureMap(
          Matrix.addPadding(m, this.filter2Size - 1),
          invertedFilterMatrixArray2[i]
        )
      );
    });
    //array length 16 - 14x14

    //STEP 5 - adjust F2
    let temp = delF2.map((m) => Matrix.multiply(m, this.learning_rate));
    this.filterMatrixArray2 = this.filterMatrixArray2.map((m, i) =>
      Matrix.subtract(m, temp[i])
    );

    //STEP 7 - add 6 filters to delP1Prime by summing delP1
    let delP1Sum = new Matrix(
      (this.inputImageSize - this.filter1Size + 1) / 2,
      (this.inputImageSize - this.filter1Size + 1) / 2
    ); //Matrix 14x14

    delP1.map((m) => delP1Sum.add(m)); //add all delP1 matrices

    let delP1Prime = []; // array of 14x14 matrices
    this.filterMatrixArray1.map((m) =>
      delP1Prime.push(
        Matrix.multiply(
          delP1Sum,
          this.filterMatrixArray1.length / this.filterMatrixArray2.length
        )
      )
    );

    //STEP 8 get delP Double Prime through reversePooling
    let delP1DoublePrime = delP1Prime.map((m, i) =>
      reversePool(m, maxPoolIndexArrayLayer1[i])
    ); //28x28x6

    //STEP 9 get delF1
    let delF1 = convolution([inputImageMatrix], delP1DoublePrime); //6 matrices

    //STEP 10 adjust F1
    let temp3 = delF1.map((m) => Matrix.multiply(m, this.learning_rate));

    this.filterMatrixArray1 = this.filterMatrixArray1.map((m, i) =>
      Matrix.subtract(m, temp3[i])
    );

    //Step 11 adjust weights of mlp
    this.mlp.train(flattenedFeatureMaps, targetArray);
  }
}
