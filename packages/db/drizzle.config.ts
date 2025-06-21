import "dotenv/config";
import type { Config } from "drizzle-kit";

export default {
  dialect: "postgresql",
  schema: "./schema.ts",
  out: "./drizzle",
  dbCredentials: {
    // Use the same connection string as in Docker Compose
    url: process.env.DATABASE_URL || "postgres://karakeep:karakeep_password@localhost:5432/karakeep",
  },
} satisfies Config;
