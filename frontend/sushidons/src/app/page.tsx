"use client";
import { MinusIcon, PlusIcon, Share2Icon, Wand2 } from "lucide-react"
import Image from "next/image"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

// Type definitions
type ProductStatus = 'in_stock' | 'sold_out' | 'coming_soon'

interface Product {
  name: string
  price: string
  image: string
  status: ProductStatus
  defaultDescription: string
  item: string
}

interface ProductsDatabase {
  [key: string]: Product
}

interface TagMetrics {
  seo_score: number
  buy_rate: number
  click_rate: number
}

interface TagGroup {
  [tag: string]: TagMetrics
}

interface TagGroups {
  category_tags?: TagGroup
  attribute_tags?: TagGroup
  style_tags?: TagGroup
  usage_tags?: TagGroup
}

interface GenerationState {
  us: boolean
  jp: boolean
}

interface TagState {
  currentLocation: string | null
  tags: {
    us: TagGroups | null
    jp: TagGroups | null
  }
  descriptions: {
    us: string | null
    jp: string | null
  }
}

interface GenerateResponse {
  tags: {
    [key: string]: TagGroups
  }
  description: string
}

interface ProductPageProps {
  productId?: string
}

interface RelatedProducts {
  [key: string]: string[]
}


// Product Data
const PRODUCTS: ProductsDatabase = {
  'kill-bill-shoe': {
    name: 'Kill Bill Shoe',
    price: '10,000.00',
    image: 'https://www.sushidon.shop/cdn/shop/files/1_540x_cc6e817d-0195-45f2-bacb-98f556172327.webp?v=1729915244&width=1426',
    status: 'in_stock',
    defaultDescription: 'Yellow and black leather shoe inspired by the movie Kill Bill.',
    item: 'shoe'
  },
  'sleeve-shirt': {
    name: 'Short Sleeve Shirt',
    price: '15,000.00',
    image: 'https://www.sushidon.shop/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426',
    status: 'in_stock',
    defaultDescription: 'Short sleeve shirt with a relaxed fit',
    item: 'shirt'
  },
  'sporty-jacket': {
    name: 'Sporty Jacket',
    price: '20,000.00',
    image: 'https://www.sushidon.shop/cdn/shop/files/453S27A4125X0853_E01-1.webp?v=1729915127&width=1426',
    status: 'in_stock',
    defaultDescription: 'Sporty jacket with a trendy fit',
    item: 'jacket'
  },
  'emb-sweater': {
    name: 'Embroidered Sweater',
    price: '20,000.00',
    image: 'https://www.sushidon.shop/cdn/shop/files/514S57A0027X8555_E01.webp?v=1729915178&width=1426',
    status: 'in_stock',
    defaultDescription: 'Grey sweater with embroidered design',
    item: 'sweater'
  }
}

const RELATED_PRODUCTS: RelatedProducts = {
  'shoe': ['kill-bill-shoe'],
  'shirt': ['sleeve-shirt'],
  'jacket': ['sporty-jacket'],
  'sweater': ['emb-sweater']
}

const PRODUCT_CATEGORIES = {
  'clothing': ['shirt', 'jacket', 'sweater','shoe'],
} as const;

export default function ProductPage({ productId = 'kill-bill-shoe' }: ProductPageProps): JSX.Element {
  const [mounted, setMounted] = useState(false)
  const [product, setProduct] = useState<Product | null>(null)
  const [quantity, setQuantity] = useState<number>(1)
  const [selectedProductId, setSelectedProductId] = useState(productId)
  const [isGenerating, setIsGenerating] = useState<GenerationState>({ us: false, jp: false })
  const [generationError, setGenerationError] = useState<string | null>(null)
  const [tagState, setTagState] = useState<TagState>({
    currentLocation: null,
    tags: { us: null, jp: null },
    descriptions: { us: null, jp: null }
  })

  useEffect(() => {
    setMounted(true)
    setProduct(PRODUCTS[selectedProductId] || PRODUCTS['kill-bill-shoe'])
  }, [selectedProductId])

  if (!mounted || !product) {
    return <div className="min-h-screen"></div>
  }

  // Update getRelatedProducts to include categorization and sorting
const getRelatedProducts = () => {
  if (!product) return [];
  
  // Get current product's category
  const currentCategory = Object.entries(PRODUCT_CATEGORIES)
    .find(([_, items]) => items.includes(product.item))?.[0];

  // Get all products sorted by relevance
  return Object.entries(PRODUCTS)
    .filter(([id]) => id !== selectedProductId) // Exclude current product
    .sort((a, b) => {
      const productA = a[1];
      const productB = b[1];

      // Same item type gets highest priority
      if (productA.item === product.item && productB.item !== product.item) return -1;
      if (productB.item === product.item && productA.item !== product.item) return 1;

      // Same category gets second priority
      const categoryA = Object.entries(PRODUCT_CATEGORIES)
        .find(([_, items]) => items.includes(productA.item))?.[0];
      const categoryB = Object.entries(PRODUCT_CATEGORIES)
        .find(([_, items]) => items.includes(productB.item))?.[0];
      
      if (categoryA === currentCategory && categoryB !== currentCategory) return -1;
      if (categoryB === currentCategory && categoryA !== currentCategory) return 1;

      return 0;
    });
};

  const generateTags = async (location: 'us' | 'jp'): Promise<void> => {
    setIsGenerating(prev => ({ ...prev, [location]: true }))
    setGenerationError(null)
    
    try {
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Origin': 'http://localhost:3000'
        },
        mode: 'cors',
        body: JSON.stringify({
          image_url: product.image,
          location: location,
          item: product.item
        })
      })
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `Failed to generate tags: ${response.statusText}`)
      }
      
      const data = await response.json()
      const productKey = Object.keys(data.tags)[0]
      const normalizedTags = data.tags[productKey] || {}

      const processedTags: TagGroups = {
        category_tags: {},
        attribute_tags: {},
        style_tags: {},
        usage_tags: {}
      }

      // Process all tag groups
      const tagGroups = ['category_tags', 'attribute_tags', 'style_tags', 'usage_tags'] as const
      tagGroups.forEach(groupName => {
        if (normalizedTags[groupName]) {
          Object.entries(normalizedTags[groupName]).forEach(([tag, value]) => {
            processedTags[groupName]![tag] = {
              seo_score: 0.8,
              buy_rate: 0.7,
              click_rate: 0.6,
              ...value
            }
          })
        }
      })

      setTagState(prev => ({
        currentLocation: location,
        tags: {
          ...prev.tags,
          [location]: processedTags
        },
        descriptions: {
          ...prev.descriptions,
          [location]: data.description
        }
      }))
      
    } catch (error) {
      console.error('Error generating tags:', error)
      setGenerationError(error instanceof Error ? error.message : 'Unknown error')
    } finally {
      setIsGenerating(prev => ({ ...prev, [location]: false }))
    }
  }

  const getBadgeVariant = (score: number): 'high' | 'average' | 'low' => {
    if (score >= 0.8) return 'high'
    if (score >= 0.4) return 'average'
    return 'low'
  }

  const handleQuantityChange = (delta: number): void => {
    const newQuantity = Math.max(1, quantity + delta)
    setQuantity(newQuantity)
  }

  const renderTags = (tagGroup: keyof TagGroups): JSX.Element | null => {
    if (!tagState.currentLocation || !tagState.tags[tagState.currentLocation]) {
      return null;
    }

    const currentTags = tagState.tags[tagState.currentLocation];
    
    if (!currentTags[tagGroup] || Object.keys(currentTags[tagGroup] || {}).length === 0) {
      return null;
    }

    return (
      <>
        {Object.entries(currentTags[tagGroup] || {}).map(([tag, metrics]) => (
          <TooltipProvider key={tag}>
            <Tooltip>
              <TooltipTrigger>
                <Badge 
                  variant={getBadgeVariant(metrics.seo_score)}
                  className="cursor-help"
                >
                  {tag.replace(/_/g, ' ')}
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                <div className="text-sm">
                  <div>SEO Score: {(metrics.seo_score * 100).toFixed(0)}%</div>
                  <div>Buy Rate: {(metrics.buy_rate * 100).toFixed(0)}%</div>
                  <div>Click Rate: {(metrics.click_rate * 100).toFixed(0)}%</div>
                </div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ))}
      </>
    )
  }

  const isOutOfStock = product.status === 'sold_out'

  return (
    <div className="grid grid-rows-[20px_1fr_20px] min-h-screen font-[family-name:var(--font-geist-sans)]">
      <main className="row-start-2 mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 w-full space-y-16">
        {/* Main Product Section */}
        <section className="grid gap-8 md:grid-cols-2">
          {/* Product Image */}
          <Card className="overflow-hidden border-0 shadow-none">
            <CardContent className="p-0">
              <div className="aspect-square relative overflow-hidden rounded-lg bg-gray-100">
                <Image
                  src={product.image}
                  alt={product.name}
                  className="object-cover object-center"
                  fill
                  priority
                />
              </div>
            </CardContent>
          </Card>
          
          {/* Product Details */}
          <div className="flex flex-col gap-4">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tight">{product.name}</h1>
              <p className="text-3xl font-bold">${product.price} USD</p>
            </div>
            
            <div className="flex items-center gap-2">
              <Badge variant="secondary">
                {isOutOfStock ? 'Sold Out' : 'In Stock'}
              </Badge>
            </div>

            {/* Quantity Selector */}
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <label htmlFor="quantity" className="font-medium">
                  Quantity
                </label>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => handleQuantityChange(-1)}
                    disabled={isOutOfStock || quantity <= 1}
                  >
                    <MinusIcon className="h-4 w-4" />
                    <span className="sr-only">Decrease quantity</span>
                  </Button>
                  <input
                    type="number"
                    id="quantity"
                    className="h-8 w-16 rounded-md border border-input bg-background px-3 py-1 text-center text-sm"
                    min="1"
                    value={quantity}
                    readOnly
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => handleQuantityChange(1)}
                    disabled={isOutOfStock}
                  >
                    <PlusIcon className="h-4 w-4" />
                    <span className="sr-only">Increase quantity</span>
                  </Button>
                </div>
              </div>

              {/* Purchase Buttons */}
              <div className="space-y-2">
                <Button className="w-full" disabled={isOutOfStock}>
                  {isOutOfStock ? 'Sold out' : 'Add to Cart'}
                </Button>
                <Button className="w-full" variant="outline">
                  Buy with ShopPay
                </Button>
                <Button variant="outline" className="w-full">
                  More payment options
                </Button>
              </div>

              {/* Tag Generation Section */}
              <div className="space-y-4">
                <div className="flex justify-end gap-2">
                  <Button
                    onClick={() => generateTags('us')}
                    disabled={isGenerating.us || isGenerating.jp}
                    variant="outline"
                    size="sm"
                    className="gap-2"
                  >
                    <Wand2 className="h-4 w-4" />
                    {isGenerating.us ? 'Generating...' : 'Generate US Tags'}
                  </Button>
                  <Button
                    onClick={() => generateTags('jp')}
                    disabled={isGenerating.us || isGenerating.jp}
                    variant="outline"
                    size="sm"
                    className="gap-2"
                  >
                    <Wand2 className="h-4 w-4" />
                    {isGenerating.jp ? 'Generating...' : 'Generate JP Tags'}
                  </Button>
                </div>

                {generationError && (
                  <p className="text-sm text-red-500">{generationError}</p>
                )}

                {/* Tags Display */}
                {isGenerating[tagState.currentLocation as keyof GenerationState] ? (
                  <div className="text-sm text-muted-foreground">Generating tags...</div>
                ) : tagState.currentLocation && tagState.tags[tagState.currentLocation as keyof TagState['tags']] ? (
                  <>
                    <div className="text-sm text-muted-foreground mb-2">
                      Showing {tagState.currentLocation.toUpperCase()} market tags
                    </div>
                    <div className="space-y-3">
                      <h3 className="font-medium">Categories</h3>
                      <div className="flex flex-wrap gap-2">
                        {renderTags('category_tags')}
                      </div>
                    </div>
                    <div className="space-y-3">
                      <h3 className="font-medium">Attributes</h3>
                      <div className="flex flex-wrap gap-2">
                        {renderTags('attribute_tags')}
                      </div>
                    </div>
                    <div className="space-y-3">
                      <h3 className="font-medium">Style</h3>
                      <div className="flex flex-wrap gap-2">
                        {renderTags('style_tags')}
                      </div>
                    </div>
                    <div className="space-y-3">
                      <h3 className="font-medium">Usage</h3>
                      <div className="flex flex-wrap gap-2">
                        {renderTags('usage_tags')}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="text-sm text-muted-foreground">
                    No tags generated yet. Click one of the generate buttons above.
                  </div>
                )}
                
                {/* Product Description */}
                <div className="space-y-4">
                  <p className="text-muted-foreground">
                    {tagState.currentLocation && 
                     tagState.descriptions[tagState.currentLocation as keyof TagState['descriptions']]
                      ? tagState.descriptions[tagState.currentLocation as keyof TagState['descriptions']]
                      : product.defaultDescription}
                  </p>

                  <Button variant="ghost" size="sm" className="gap-2">
                    <Share2Icon className="h-4 w-4" />
                    Share
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Related Products Section */}
        <section className="mt-16 border-t pt-16">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold tracking-tight">You might also like</h2>
            {product.item !== 'all' && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  // Optional: add filtering logic here
                }}
              >
                View All
              </Button>
            )}
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6">
            {getRelatedProducts().map(([id, prod]) => (
              <Card 
                key={id}
                className={`overflow-hidden cursor-pointer transition-all duration-200 hover:scale-105 ${
                  id === selectedProductId ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => setSelectedProductId(id)}
              >
                <CardContent className="p-0">
                  <div className="aspect-square relative overflow-hidden rounded-lg bg-gray-100">
                    <Image
                      src={prod.image}
                      alt={prod.name}
                      className="object-cover object-center"
                      fill
                      sizes="(max-width: 768px) 50vw, (max-width: 1024px) 33vw, 25vw"
                    />
                  </div>
                  <div className="p-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium truncate">{prod.name}</h3>
                        <p className="text-sm text-muted-foreground mt-1">
                          ${prod.price} USD
                        </p>
                      </div>
                      <Badge variant="outline" className="capitalize">
                        {prod.item}
                      </Badge>
                    </div>
                    {prod.status === 'sold_out' && (
                      <Badge variant="secondary" className="mt-2">
                        Sold Out
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center py-4">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="#"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/file.svg"
            alt="File icon"
            width={16}
            height={16}
          />
          Shipping Info
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="#"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/window.svg"
            alt="Window icon"
            width={16}
            height={16}
          />
          Size Guide
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="#"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/globe.svg"
            alt="Globe icon"
            width={16}
            height={16}
          />
          Returns Policy →
        </a>
      </footer>
    </div>
  )
}