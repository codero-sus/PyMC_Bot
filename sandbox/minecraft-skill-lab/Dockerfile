# Operator supplies the official 26.1 Java server jar; it is not distributed here.
FROM eclipse-temurin:25-jre AS java
FROM node:22-bookworm-slim
LABEL org.pymcbot.skill-lab="1"
COPY --from=java /opt/java/openjdk /opt/java/openjdk
ENV PATH="/opt/java/openjdk/bin:${PATH}"
WORKDIR /lab
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts --no-audit --no-fund
COPY server.jar /lab/server.jar
RUN echo '3872a7f07a1a595e651aef8b058dfc2bb3772f46  /lab/server.jar' | sha1sum -c
COPY runner.js worker.js /lab/
USER 10001:10001
ENTRYPOINT ["node", "/lab/runner.js"]
