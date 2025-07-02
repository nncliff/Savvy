import { eq } from "drizzle-orm";
import { DequeuedJob, Runner } from "liteque";

import type { ZOpenAIRequest } from "@karakeep/shared/queues";
import { db } from "@karakeep/db";
import { bookmarks } from "@karakeep/db/schema";
import serverConfig from "@karakeep/shared/config";
import { InferenceClientFactory } from "@karakeep/shared/inference";
import logger from "@karakeep/shared/logger";
import { OpenAIQueue, zOpenAIRequestSchema } from "@karakeep/shared/queues";

import { runSummarization } from "./summarize";
import { runTagging } from "./tagging";

async function attemptMarkStatus(
  jobData: object | undefined,
  status: "success" | "failure",
) {
  if (!jobData) {
    return;
  }
  try {
    const request = zOpenAIRequestSchema.parse(jobData);
    await db
      .update(bookmarks)
      .set({
        ...(request.type === "summarize"
          ? { summarizationStatus: status }
          : {}),
        ...(request.type === "tag" ? { taggingStatus: status } : {}),
      })
      .where(eq(bookmarks.id, request.bookmarkId));
  } catch (e) {
    logger.error(`Something went wrong when marking the tagging status: ${e}`);
  }
}

export class OpenAiWorker {
  static build() {
    logger.info("========== INFERENCE WORKER INITIALIZATION ==========");
    logger.info("Starting inference worker ...");
    logger.info(`Inference config - openAIApiKey: ${serverConfig.inference.openAIApiKey ? 'SET' : 'NOT SET'}`);
    logger.info(`Inference config - ollamaBaseUrl: ${serverConfig.inference.ollamaBaseUrl || 'NOT SET'}`);
    logger.info(`Inference config - enableAutoTagging: ${serverConfig.inference.enableAutoTagging}`);
    logger.info(`Inference config - enableAutoSummarization: ${serverConfig.inference.enableAutoSummarization}`);
    logger.info(`Inference config - jobTimeoutSec: ${serverConfig.inference.jobTimeoutSec}`);
    
    const worker = new Runner<ZOpenAIRequest>(
      OpenAIQueue,
      {
        run: runOpenAI,
        onComplete: async (job) => {
          const jobId = job.id;
          logger.info(`[inference][${jobId}] ✅ Job completed successfully`);
          await attemptMarkStatus(job.data, "success");
        },
        onError: async (job) => {
          const jobId = job.id;
          logger.error(
            `[inference][${jobId}] ❌ Job failed: ${job.error}\n${job.error.stack}`,
          );
          if (job.numRetriesLeft == 0) {
            logger.error(`[inference][${jobId}] No retries left, marking as failure`);
            await attemptMarkStatus(job?.data, "failure");
          } else {
            logger.warn(`[inference][${jobId}] Job will be retried. Retries left: ${job.numRetriesLeft}`);
          }
        },
      },
      {
        concurrency: 1,
        pollIntervalMs: 1000,
        timeoutSecs: serverConfig.inference.jobTimeoutSec,
      },
    );

    logger.info("Inference worker initialized successfully with:");
    logger.info(`  - Concurrency: 1`);
    logger.info(`  - Poll interval: 1000ms`);
    logger.info(`  - Timeout: ${serverConfig.inference.jobTimeoutSec}s`);
    logger.info("========== INFERENCE WORKER READY ==========");
    
    return worker;
  }
}

async function runOpenAI(job: DequeuedJob<ZOpenAIRequest>) {
  const jobId = job.id;
  logger.info(`[inference][${jobId}] 🚀 NEW JOB RECEIVED`);
  logger.info(`[inference][${jobId}] Job details: ${JSON.stringify(job.data, null, 2)}`);

  logger.info(`[inference][${jobId}] Checking inference client configuration...`);
  const inferenceClient = InferenceClientFactory.build();
  if (!inferenceClient) {
    logger.warn(
      `[inference][${jobId}] ⚠️  No inference client configured - skipping job`,
    );
    logger.warn(`[inference][${jobId}] Check that either OPENAI_API_KEY or OLLAMA_BASE_URL is set`);
    logger.warn(`[inference][${jobId}] Current config - openAIApiKey: ${serverConfig.inference.openAIApiKey ? 'SET' : 'NOT SET'}`);
    logger.warn(`[inference][${jobId}] Current config - ollamaBaseUrl: ${serverConfig.inference.ollamaBaseUrl || 'NOT SET'}`);
    return;
  }

  logger.info(`[inference][${jobId}] ✅ Inference client found and configured`);
  logger.info(`[inference][${jobId}] Parsing job request...`);

  const request = zOpenAIRequestSchema.safeParse(job.data);
  if (!request.success) {
    logger.error(`[inference][${jobId}] ❌ Malformed job request`);
    logger.error(`[inference][${jobId}] Parse error: ${request.error.toString()}`);
    throw new Error(
      `[inference][${jobId}] Got malformed job request: ${request.error.toString()}`,
    );
  }

  logger.info(`[inference][${jobId}] ✅ Job request parsed successfully`);
  const { bookmarkId } = request.data;
  const jobType = request.data.type;
  
  logger.info(`[inference][${jobId}] Processing job:`);
  logger.info(`[inference][${jobId}]   - Type: ${jobType}`);
  logger.info(`[inference][${jobId}]   - Bookmark ID: ${bookmarkId}`);

  switch (request.data.type) {
    case "summarize":
      logger.info(`[inference][${jobId}] 📝 Delegating to summarization handler...`);
      await runSummarization(bookmarkId, job, inferenceClient);
      break;
    case "tag":
      logger.info(`[inference][${jobId}] 🏷️  Delegating to tagging handler...`);
      await runTagging(bookmarkId, job, inferenceClient);
      break;
    default:
      logger.error(`[inference][${jobId}] ❌ Unknown job type: ${request.data.type}`);
      throw new Error(`Unknown inference type: ${request.data.type}`);
  }
  
  logger.info(`[inference][${jobId}] Job handler completed, returning to worker...`);
}
