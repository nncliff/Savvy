import { db } from "./drizzle";
import { migrate } from "drizzle-orm/postgres-js/migrator";

async function main() {
  try {
    await migrate(db, { migrationsFolder: "./drizzle" });
    console.log("Migration completed");
    process.exit(0);
  } catch (err) {
    console.error("Migration failed:", err);
    process.exit(1);
  }
}

main();
