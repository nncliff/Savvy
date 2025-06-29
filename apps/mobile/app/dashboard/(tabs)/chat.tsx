import React, { useState, useRef, useEffect } from "react";
import { View, Text, TextInput, TouchableOpacity, ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView } from "react-native";
import useAppSettings from "@/lib/settings";

export default function ChatScreen() {
  const { settings, isLoading: isSettingsLoading } = useAppSettings();
  const [messages, setMessages] = useState([
    { role: "system", content: "To discuss about your bookmarks." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollViewRef = useRef<ScrollView | null>(null);

  useEffect(() => {
    if (scrollViewRef.current) {
      scrollViewRef.current.scrollToEnd({ animated: true });
    }
  }, [messages]);

  async function handleSend() {
    if (!input.trim() || isSettingsLoading || !settings.address) return;
    setLoading(true);
    setMessages((msgs) => [...msgs, { role: "user", content: input }]);
    try {
      const res = await fetch(`${settings.address}/api/llamaindex`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input }),
      });
      let data = null;
      try {
        data = await res.json();
      } catch (jsonErr) {
        // Ignore JSON parse errors
      }
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      setMessages((msgs) => [
        ...msgs,
        { role: "assistant", content: data?.response || data?.error || "No response." },
      ]);
    } catch (e) {
      setMessages((msgs) => [
        ...msgs,
        { role: "assistant", content: `Error contacting server: ${e instanceof Error ? e.message : e}` },
      ]);
    }
    setInput("");
    setLoading(false);
  }

  function handleKeyDown(e: { nativeEvent: { key: string } }) {
    if (e.nativeEvent.key === "Enter" && !loading) handleSend();
  }

  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: "#fff" }}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      keyboardVerticalOffset={80}
    >
      <View style={{ flex: 1, padding: 16 }}>
        <ScrollView
          ref={scrollViewRef}
          contentContainerStyle={{ paddingBottom: 16 }}
          onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
        >
          {messages.map((msg, i) => (
            <View
              key={i}
              style={{
                alignSelf:
                  msg.role === "user"
                    ? "flex-end"
                    : msg.role === "assistant"
                    ? "flex-start"
                    : "center",
                backgroundColor:
                  msg.role === "user"
                    ? "#e0e7ff"
                    : msg.role === "assistant"
                    ? "#dcfce7"
                    : "#f3f4f6",
                borderRadius: 12,
                marginVertical: 4,
                padding: 10,
                maxWidth: "80%",
              }}
            >
              <Text style={{ color: msg.role === "user" ? "#3730a3" : msg.role === "assistant" ? "#166534" : "#6b7280" }}>
                {msg.content}
              </Text>
            </View>
          ))}
        </ScrollView>
        <View style={{ flexDirection: "row", alignItems: "center", marginTop: 8 }}>
          <TextInput
            style={{
              flex: 1,
              borderWidth: 1,
              borderColor: "#d1d5db",
              borderRadius: 8,
              paddingHorizontal: 12,
              paddingVertical: 8,
              marginRight: 8,
              backgroundColor: loading ? "#f3f4f6" : "#fff",
            }}
            placeholder="Type your question..."
            value={input}
            onChangeText={setInput}
            onSubmitEditing={handleSend}
            editable={!loading}
            returnKeyType="send"
            blurOnSubmit={false}
          />
          <TouchableOpacity
            onPress={handleSend}
            disabled={loading}
            style={{
              backgroundColor: loading ? "#93c5fd" : "#2563eb",
              paddingHorizontal: 16,
              paddingVertical: 10,
              borderRadius: 8,
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={{ color: "#fff", fontWeight: "bold" }}>Send</Text>
            )}
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}
