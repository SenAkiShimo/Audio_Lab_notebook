import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.CompletableFuture;

public class SecurityGateway {
    private static final int MAX_ALLOWABLE_DURATION_SECONDS = 30;
    private static final int REQUIRED_BIT_DEPTH = 16;
    private static final int ALLOWED_SAMPLING_RATE_16K = 16000;

    public CompletableFuture<Boolean> validateStreamAsync(byte[] rawByteStream) {
        return CompletableFuture.supplyAsync(() -> {
            if (rawByteStream == null || rawByteStream.length < 44) return false;

            ByteBuffer headerReader = ByteBuffer.wrap(rawByteStream, 0, 44).order(ByteOrder.LITTLE_ENDIAN);
            int sampleRate = headerReader.getInt(24);   
            short bitDepth = headerReader.getShort(34); 

            if (sampleRate != ALLOWED_SAMPLING_RATE_16K || bitDepth != REQUIRED_BIT_DEPTH) return false;

            long totalPcmBytes = rawByteStream.length - 44;
            double computedDuration = (double) totalPcmBytes / 32000.0;

            return computedDuration <= MAX_ALLOWABLE_DURATION_SECONDS;
        });
    }

    public static void main(String[] args) throws Exception {
        System.out.println("============================================================");
        System.out.println("[RUN]");

        SecurityGateway gateway = new SecurityGateway();

        // 1. A simulated hacker uploaded an extremely massive, malicious data stream—designed to overwhelm the server's memory—in a simulated Denial-of-Service (DoS) attack.
        byte[] malicious_dos_stream = new byte[32000 * 50 + 44];
        ByteBuffer buf = ByteBuffer.wrap(malicious_dos_stream).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(24, 16000); // Insert compliant sample rate
        buf.putShort(34, (short)16); // Insert compliant bit depth

        // 2. Route into the high-concurrency, asynchronous, non-blocking security validation pipeline.
        CompletableFuture<Boolean> result = gateway.validateStreamAsync(malicious_dos_stream);
        
        // 3. Block and wait for the validation result, observing the gateway's circuit breaker.
        boolean pass = result.get();
        System.out.println("[GATEWAY VERDICT] Validation complete. Approval status: " + (pass ? "Approved (PASS)" : "🚨 Blocked and tripped the connection (BLOCKED)"));
        System.out.println("============================================================");
    }
}
