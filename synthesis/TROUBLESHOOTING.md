# Troubleshooting

## Service Does Not Start

Check credentials first:

```bash
cat .env
docker compose logs synthesis
```

Common causes:

- missing `MODEL_API_PROVIDER`
- missing `OPENAI_API_KEY`
- missing Azure endpoint or deployment settings

## `401 Unauthorized`

Your provider credentials are wrong or expired.

Check:

- `OPENAI_API_KEY`
- or the Azure endpoint, key, and deployment names

## `429 Too Many Requests`

Reduce load in `.env`:

```env
REQUESTS_PER_MINUTE=5
MAX_CONCURRENT=1
```

## Retrieval Images Fail To Stage

If `image_urls` like `"/images/..."` fail, verify:

```bash
curl http://localhost:8000/health
```

Also check `SYNTHESIS_RETRIEVAL_BASE_URL`. Inside Docker Compose it defaults to `http://host.docker.internal:8000`.

## `No .jpg/.jpeg/.png images found`

Either:

- put seed images in `input-images/`
- pass `image_urls`
- upload files with multipart form data

## A Job Needs To Be Stopped

```bash
curl -X POST http://localhost:8090/jobs/<job_id>/cancel
```

## Security Note

`docker compose config` expands environment variables and prints secrets. Do not paste that output into shared logs unless you have rotated the keys.
