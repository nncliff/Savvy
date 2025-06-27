"use client";
import dynamic from "next/dynamic";

// Dynamically import the chat so it only renders on the client
const RightRailChat = dynamic(() => import("../../../../landing/src/RightRailChat"), { ssr: false });

export default function RightRailChatClient() {
  return <RightRailChat />;
}
