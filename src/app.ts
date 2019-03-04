import * as tf from "@tensorflow/tfjs"

// import "@tensorflow/tfjs-node-gpu"
import "@tensorflow/tfjs-node"

import { ImageData, IBatch } from "./data"

import { model } from "./network"

const TEST_ITERATION_FREQUENCY = 5 // how often to update the validation data

async function testRun(data: ImageData) {
  const tensors = await data.getTestBatch()
  const tidy = tf.tidy(() => {
    const resultsTensor: any = model.predict(tensors.tensors)
    const expected: any = Array.from<number>(tensors.labels.dataSync())
    const results = Array.from<number>(resultsTensor.dataSync())
    const blah = expected.map((expect, i) => {
      // return { expected: expect, actual: (expect - results[i]) ** 2 }
      return { expected: expect, actual: results[i] }
    })
    // console.log(results)
    console.log(blah) // 1 good, 0 bad
    // blah.sort((a, b) => a[0] - b[0]).forEach(console.log)
  })
}

async function trainRun(data: ImageData) {
  let step = 0

  console.log("\x1b[33m%s\x1b[0m", "====== step =====", step)
  let batch: IBatch | undefined = await data.nextBatch()

  while (batch) {
    console.time("training batch")
    let validationData
    const { tensors, labels, batchSize } = batch

    if (step % TEST_ITERATION_FREQUENCY === 0) {
      console.log("> Loading validations")
      const validationBatch = await data.getValidationBatch()
      validationData = [validationBatch.tensors, validationBatch.labels]
    }

    console.time("fitting")
    await model.fit(tensors, labels, {
      batchSize,
      validationData,
      epochs: 1,
      verbose: 0,
    })
    console.timeEnd("fitting")

    tf.dispose([tensors, labels, validationData])

    console.timeEnd("training batch")
    console.log("\x1b[33m%s\x1b[0m", "====== step =====", step)
    batch = await data.nextBatch()
    // console.log(tf.memory())
    step++
  }
}
async function testImage(data: ImageData) {
  const tensor = await data.loadTestImage()
  const tidy = tf.tidy(() => {
    const temp: any = model.predict(tensor)
    temp.print()
  })
}

async function run() {
  const data = new ImageData({
    batchSize: 32,
    validationDataPerc: 0.2,
    testDataPerc: 0.1,
  })
  console.log("Training Network")
  await trainRun(data)
  console.log("Testing Network")
  await testRun(data)
  console.log("Saving Model")
  await model.save("file://src/model/tempModel")
  console.log("Testing Image")
}

async function loadOne() {
  console.log("\x1b[31m", "--------Load One--------")
  const data = new ImageData({})
  const loadedModel = await tf.loadModel(
    "file://src/model/tempModel/model.json"
  )
  console.log("\x1b[37m", "Testing Image")
  const tensor = await data.loadTestImage()
  const tidy = tf.tidy(() => {
    const temp: any = loadedModel.predict(tensor)
    const results = Array.from<number>(temp.dataSync())
    temp.print()
    console.log({
      expected: [0.1, 0.1, 0.9],
      actual: results,
    })
    console.log({
      expected: 0.1,
      actual: [
        (0.1 - results[0]) ** 2,
        (0.1 - results[1]) ** 2,
        (0.1 - results[2]) ** 2,
      ],
    })
  })
}

async function loadTwo() {
  let val1
  let val2
  console.log("\x1b[31m", "--------vs Apple Ceder Rust--------")
  const data = new ImageData({})
  const loadedModel = await tf.loadModel(
    "file://src/model/appleCedarRust/model.json"
  )
  console.log("\x1b[37m", "Testing Image")
  const tensor = await data.loadTestImage()
  const tidy = tf.tidy(() => {
    const temp: any = loadedModel.predict(tensor)
    const results = Array.from<number>(temp.dataSync())
    console.log({
      expected: 0.1,
      actual: results,
    })
    console.log({
      expected: 0.1,
      actual: (0.1 - results[0]) ** 2,
    })
    val1 = results
  })
  console.log("\x1b[31m", "--------vs Apple Scab--------")
  const data2 = new ImageData({})
  const loadedModel2 = await tf.loadModel(
    "file://src/model/appleScab/model.json"
  )
  console.log("\x1b[37m", "Testing Image")
  const tensor2 = await data2.loadTestImage()
  const tidy2 = tf.tidy(() => {
    const temp2: any = loadedModel2.predict(tensor2)
    const results2 = Array.from<number>(temp2.dataSync())
    console.log({
      expected: 0.1,
      actual: results2,
    })
    console.log({
      expected: 0.1,
      actual: (0.1 - results2[0]) ** 2,
    })
    val2 = results2
  })
  console.log("\x1b[32m", "softmax", "\x1b[37m")
  const a = tf.tensor1d([val1[0], val2[0]])
  a.softmax().print()
}

// run()
loadOne()
// loadTwo()
