import osmnx as ox

# tìm tất cả điểm bán grocery/convenience trong quận 1
tags = {"shop": ["grocery", "convenience", "supermarket"]}
gdf = ox.geometries_from_place("District 1, Ho Chi Minh City, Vietnam", tags)

#print(gdf[["shop", "name", "geometry"]].head())
print(gdf[["shop", "name", "geometry"].count()])
