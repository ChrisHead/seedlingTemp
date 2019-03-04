import * as fs from "fs"
import pgPromise from "pg-promise"
import { promisify } from "util"

import { PlantNetwork } from "./src/PlantNetwork"
import { IClassificationResult } from "./src/types"
const readFile = promisify(fs.readFile)

const pgp = pgPromise()
const conn = pgp({
  user: "postgres",
  host: "localhost",
  database: "seedling_development",
  password: "pass123",
  port: 5432,
})

interface ITicketRow {
  id: string
  image_path: string
  status: string
  user_id: string | null
  result: IClassificationResult | null
}

async function run() {
  console.log("Checking...")
  const ticket: ITicketRow | undefined = await conn.oneOrNone(
    "SELECT * FROM tickets WHERE status='pending' LIMIT 1;"
  )
  if (!ticket) {
    console.log("No Pending Tickets")
    setTimeout(run, 5000)
    return
  }
  console.log("Classifying")
  const file = await readFile(ticket.image_path)
  const network = await PlantNetwork.load()
  const result = await network.classify(file)
  console.log(
    "Classified - ",
    `${result.final.type}@${result.final.confidence}`
  )
  conn.none(
    `UPDATE tickets SET status='complete', result=$<result:json>, classified_at=now() WHERE id=$<id>`,
    {
      id: ticket.id,
      result,
    }
  )

  setTimeout(run, 1000)
}

run()
