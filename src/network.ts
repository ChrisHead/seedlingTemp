import "@tensorflow/tfjs-node"

import * as tf from "@tensorflow/tfjs"

const model = tf.sequential()

model.add(
  tf.layers.conv2d({
    inputShape: [256, 256, 3],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: "sigmoid",
    kernelInitializer: "VarianceScaling",
  })
)

model.add(
  tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2],
  })
)

model.add(
  tf.layers.conv2d({
    inputShape: [126, 126, 8],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: "sigmoid",
    kernelInitializer: "VarianceScaling",
  })
)

model.add(
  tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2],
  })
)

model.add(tf.layers.flatten())

// model.add(
//   tf.layers.dense({
//     units: 1000,
//     kernelInitializer: "VarianceScaling",
//     activation: "sigmoid",
//   })
// )

model.add(
  tf.layers.dense({
    units: 3,
    kernelInitializer: "VarianceScaling",
    activation: "sigmoid",
  })
)

const LEARNING_RATE = 0.01
const optimizer = tf.train.sgd(LEARNING_RATE)

model.compile({
  optimizer,
  loss: "meanSquaredError",
  metrics: ["mse"],
  // loss: "categoricalCrossentropy",
  // metrics: ["accuracy"],
})

export { model }
