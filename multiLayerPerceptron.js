//sigmoid functions introduce non-linearity
function sigmoid(x) {
  // return 1 / (1 + Math.exp(-x));
  return x > 0 ? x : 0;
}
function dsigmoid(y) {
  // return y * (1 - y);
  return y > 0 ? 1 : 0.001;
}

//the class implements feedforward and backpropagation for an mlp with layer dimensions specified by the user
//backpropagation fine-tunes the internal weights and biases and shifts them towards a local minimum to reduce prediction error
class MultiLayerPerceptron {
  //takes in
  constructor(array) {
    this.learning_rate = 0.5;

    //store node lengths
    this.node_lengths = array;
    //initialize weights and bias matrix arrays
    this.weights_array = [];
    this.bias_array = [];
    //store outputs from each layer in vertical Matrices, updated each time prediction is made, not when trained
    this.outputs_array = []; //array of one column matrices, length is node length - 1

    for (let i = 0; i < array.length - 1; i++) {
      let weightMatrix = new Matrix(array[i + 1], array[i]);
      weightMatrix.randomize();
      this.weights_array.push(weightMatrix);

      let biasMatrix = new Matrix(array[i + 1], 1);
      biasMatrix.randomize();
      this.bias_array.push(biasMatrix);
    }
  }

  predict(input_array) {
    let input = Matrix.fromArray(input_array);

    this.outputs_array = [];
    //Generate array of outputs
    for (let i = 0; i < this.node_lengths.length - 1; i++) {
      let output =
        i == 0
          ? Matrix.cross(this.weights_array[i], input)
          : Matrix.cross(this.weights_array[i], this.outputs_array[i - 1]);
      output.add(this.bias_array[i]);
      output.map(sigmoid);
      this.outputs_array.push(output);
    }
    //Send back to caller - REFACTOR
    // return this.outputs_array.map((m) => Matrix.toArray(m));
    return Matrix.toArray(this.outputs_array[this.outputs_array.length - 1]);
  }

  train(input_array, target_array) {
    let input = Matrix.fromArray(input_array);
    let target = Matrix.fromArray(target_array);

    let outputsArray = []; //store outputs at each layer
    for (let i = 0; i < this.node_lengths.length - 1; i++) {
      let output =
        i == 0
          ? Matrix.cross(this.weights_array[i], input)
          : Matrix.cross(this.weights_array[i], outputsArray[i - 1]);
      output.add(this.bias_array[i]);
      output.map(sigmoid);
      outputsArray.push(output);
    }

    //Generate Errors
    let errorMatrix = [];
    for (let i = 0; i < this.node_lengths.length - 1; i++) {
      //Get error
      let error =
        i == 0
          ? Matrix.subtract(target, outputsArray[outputsArray.length - 1 - i])
          : Matrix.cross(
              Matrix.transpose(this.weights_array[outputsArray.length - i]),
              errorMatrix[i - 1]
            );

      errorMatrix.push(error);
      //Calculate gradient Y(1-Y)
      let gradient = Matrix.map(
        outputsArray[outputsArray.length - 1 - i],
        dsigmoid
      );

      gradient.multiply(error);
      gradient.multiply(this.learning_rate);

      //Calculate DELTA W and DELTA B
      let temp = Matrix.transpose(
        [input, ...outputsArray][outputsArray.length - 1 - i]
      ); //PROBLEM
      let weights_delta = Matrix.cross(gradient, temp);

      this.weights_array[outputsArray.length - 1 - i].add(weights_delta);
      this.bias_array[outputsArray.length - 1 - i].add(gradient);
    }
  }
}
