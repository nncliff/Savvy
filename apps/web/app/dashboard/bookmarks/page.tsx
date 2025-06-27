import React from "react";
import Bookmarks from "@/components/dashboard/bookmarks/Bookmarks";
import RightRailChatClient from "./RightRailChatClient";

export default async function BookmarksPage() {
  return (
    <div>
      <Bookmarks query={{ archived: false }} showEditorCard={true} />
      <RightRailChatClient />
    </div>
  );
}
