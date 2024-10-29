from google.analytics.data import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunRealtimeReportRequest

key_path = './config.json'
client = BetaAnalyticsDataClient.from_service_account_file(key_path)
property_id = '464939780'

request = RunRealtimeReportRequest(
    property=f'properties/{property_id}',
    dimensions=[{"name": "platform"}],
    metrics=[{"name": "activeUsers"}]
)

response = client.run_realtime_report(request)

for row in response.rows:
    platform = row.dimension_values[0].value
    active_users = row.metric_values[0].value
    print(f'Platform: {platform}, Active Users: {active_users}')