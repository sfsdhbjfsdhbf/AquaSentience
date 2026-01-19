package com.ywy.ywyagent.context;


public class ChatIdContext {
    private static final ThreadLocal<String> CHAT_ID = new ThreadLocal<>();

    /**
     * 设置当前线程的 chatId
     *
     * @param chatId 聊天室ID
     */
    public static void set(String chatId) {
        CHAT_ID.set(chatId);
    }

    /**
     * 获取当前线程的 chatId
     *
     * @return chatId，如果未设置则返回 null
     */
    public static String get() {
        return CHAT_ID.get();
    }

    /**
     * 清除当前线程的 chatId
     * 在请求处理完成后调用，避免内存泄漏
     */
    public static void clear() {
        CHAT_ID.remove();
    }
}