from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest
from google.oauth2 import service_account

key_path = './config.json'
credentials = service_account.Credentials.from_service_account_file(key_path)
client = BetaAnalyticsDataClient(credentials=credentials)
property_id = '464939780'

request = RunReportRequest(
    property=f'properties/{property_id}',
    # dimensions=[{"name": "city"}],
    dimensions=[{"name": "searchTerm"}],
    # metrics=[{"name": "activeUsers"}, 
    #          {"name": "date"}],
    metrics=[{"name": "sessions"}, {"name": "engagementRate"}],
    date_ranges=[{"start_date": "2024-01-01", "end_date": "2024-10-28"}]
)

response = client.run_report(request)

for row in response.rows:
    # print([dimension.value for dimension in row.dimension_values],
    #       [metric.value for metric in row.metric_values])
    search_term = row.dimension_values[0].value
    sessions = row.metric_values[0].value
    engagement_rate = row.metric_values[1].value
    print(f'Search Term: {search_term}, Sessions: {sessions}, Engagement Rate: {engagement_rate}')
    
'''
potentially useful?
itemBrand	Item brand	Brand name of the item.
itemCategory	Item category	The hierarchical category in which the item is classified. For example, in Apparel/Mens/Summer/Shirts/T-shirts, Apparel is the item category.
itemCategory2	Item category 2	The hierarchical category in which the item is classified. For example, in Apparel/Mens/Summer/Shirts/T-shirts, Mens is the item category 2.
itemCategory3	Item category 3	The hierarchical category in which the item is classified. For example, in Apparel/Mens/Summer/Shirts/T-shirts, Summer is the item category 3.
itemCategory4	Item category 4	The hierarchical category in which the item is classified. For example, in Apparel/Mens/Summer/Shirts/T-shirts, Shirts is the item category 4.
itemCategory5	Item category 5	The hierarchical category in which the item is classified. For example, in Apparel/Mens/Summer/Shirts/T-shirts, T-shirts is the item category 5.
itemId	Item ID	The ID of the item.
itemName	Item name	The name of the item.
percentScrolled	Percent scrolled	The percentage down the page that the user has scrolled (for example, 90). Automatically populated if Enhanced Measurement is enabled. Populated by the event parameter percent_scrolled.
organicGoogleSearchAveragePosition	Organic Google Search average position	The average ranking of your website URLs for the query reported from Search Console. For example, if your site's URL appears at position 3 for one query and position 7 for another query, the average position would be 5 (3+7/2). This metric requires an active Search Console link.
organicGoogleSearchClickThroughRate	Organic Google Search click through rate	The organic Google Search click through rate reported from Search Console. Click through rate is clicks per impression. This metric is returned as a fraction; for example, 0.0588 means about 5.88% of impressions resulted in a click. This metric requires an active Search Console link.
organicGoogleSearchClicks	Organic Google Search clicks	The number of organic Google Search clicks reported from Search Console. This metric requires an active Search Console link.
organicGoogleSearchImpressions	Organic Google Search impressions	The number of organic Google Search impressions reported from Search Console. This metric requires an active Search Console link.
firstUserGoogleAdsKeyword	First user Google Ads keyword text	First user Google Ads keyword text
firstUserGoogleAdsQuery	First user Google Ads query	The search query that first acquired the user.
'''