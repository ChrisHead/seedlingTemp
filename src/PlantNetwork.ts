import { IClassificationResult } from "./types"

interface INetwork {
  name: string
  forward(img: any): Promise<number>
}
export class PlantNetwork {
  static load() {
    return new PlantNetwork()
  }

  networks: INetwork[] = [
    {
      name: "apple",
      async forward(img: any) {
        return 0.1
      },
    },
    {
      name: "banana",
      async forward(img: any) {
        return 0.3
      },
    },
    {
      name: "peach",
      async forward(img: any) {
        return 0.8
      },
    },
  ]

  async classify(image: any): Promise<IClassificationResult> {
    const resultProms = this.networks.map(network => network.forward(image))
    const results = await Promise.all(resultProms)

    const raw: IClassificationResult["raw"] = results.reduce(
      (acc, result, i) => {
        acc[this.networks[i].name] = result
        return acc
      },
      {} as any
    )

    const final = this.chooseFromRaw(raw)

    return { raw, final }
  }

  chooseFromRaw(raw: IClassificationResult["raw"]): IClassificationResult["final"] {
    const chosen = Object.keys(raw).reduce(
      (acc, el) => {
        if (raw[el] > acc.confidence) {
          return { type: el, confidence: raw[el] }
        } else {
          return acc
        }
      },
      { type: "", confidence: -Infinity }
    )
    return chosen
  }
}
