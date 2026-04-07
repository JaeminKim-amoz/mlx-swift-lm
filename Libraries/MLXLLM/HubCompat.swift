// Backward-compatibility shim: restores the pre-#118 convenience API
// (`loadModelContainer(id:progressHandler:)`) using HubApi / AutoTokenizer
// from swift-transformers so existing callers don't need to change.

import Foundation
import Hub
import MLXLMCommon
import Tokenizers

// MARK: - HuggingFace Downloader bridge

private struct HubDownloader: MLXLMCommon.Downloader, @unchecked Sendable {
    let hub: HubApi

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        try await hub.snapshot(
            from: id,
            revision: revision ?? "main",
            matching: patterns,
            progressHandler: progressHandler
        )
    }
}

// MARK: - HuggingFace TokenizerLoader bridge

private struct HubTokenizerLoader: MLXLMCommon.TokenizerLoader, @unchecked Sendable {
    let hub: HubApi

    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory, hubApi: hub)
        return TokenizerBridge(upstream)
    }
}

private struct TokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

// MARK: - Convenience overloads (pre-#118 API)

/// Load a model by HuggingFace repo ID, downloading via the default HubApi.
/// Backward-compatible replacement for the pre-#118 convenience API.
public func loadModelContainer(
    hub: HubApi = .shared,
    id: String,
    revision: String = "main",
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> MLXLMCommon.ModelContainer {
    try await MLXLMCommon.loadModelContainer(
        from: HubDownloader(hub: hub),
        using: HubTokenizerLoader(hub: hub),
        id: id,
        revision: revision,
        progressHandler: progressHandler
    )
}
